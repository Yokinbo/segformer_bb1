import os
import time

import numpy as np
import torch
from tqdm import tqdm

from multispectral_config import band_mode, image_ext, in_channels, selected_bands
from segformer import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results


MODEL_NAME = "SegFormer"
EFFICIENCY_WARMUP_ITERS = 10
EFFICIENCY_BENCHMARK_ITERS = 100


def synchronize_device(device):
    """CUDA 异步执行，计时前后同步后才能得到真实的模型前向耗时。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_model_flops(model, dummy_input):
    """使用 PyTorch Profiler 统计一次前向传播的 FLOPs。"""
    profiler = getattr(torch, "profiler", None)
    if profiler is None:
        raise RuntimeError("当前 PyTorch 不支持 torch.profiler，无法统计 FLOPs。")

    synchronize_device(dummy_input.device)
    with torch.no_grad():
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU],
            with_flops=True,
        ) as profile_result:
            model(dummy_input)
    synchronize_device(dummy_input.device)

    total_flops = sum(float(event.flops or 0.0) for event in profile_result.key_averages())
    if total_flops <= 0:
        raise RuntimeError("PyTorch Profiler 未统计到有效 FLOPs。")
    return total_flops


def benchmark_model_efficiency(model, input_shape, device, warmup_iters, benchmark_iters):
    """统计参数量、FLOPs、batch=1 模型前向延迟和 FPS。"""
    if warmup_iters < 0:
        raise ValueError("warmup_iters 不能小于 0")
    if benchmark_iters <= 0:
        raise ValueError("benchmark_iters 必须大于 0")

    model.eval()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    dummy_input = torch.zeros(
        (1, in_channels, int(input_shape[0]), int(input_shape[1])),
        dtype=torch.float32,
        device=device,
    )
    total_flops = count_model_flops(model, dummy_input)

    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        with torch.no_grad():
            for _ in range(warmup_iters):
                model(dummy_input)
            synchronize_device(device)

            start = time.perf_counter()
            for _ in range(benchmark_iters):
                model(dummy_input)
            synchronize_device(device)
            elapsed_seconds = time.perf_counter() - start
    finally:
        torch.backends.cudnn.benchmark = original_cudnn_benchmark

    latency_ms = elapsed_seconds * 1000.0 / benchmark_iters
    return {
        "params_m": float(total_params / 1e6),
        "flops_g": float(total_flops / 1e9),
        "latency_ms_per_image": float(latency_ms),
        "fps": float(1000.0 / latency_ms),
        "warmup_iters": int(warmup_iters),
        "benchmark_iters": int(benchmark_iters),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def paper_band_label(value):
    return {
        "rgb": "rgb",
        "4band": "4bands",
        "6band": "6bands",
    }.get(value, value)


def save_accuracy_efficiency_txt(output_dir, IoUs, PA_Recall, Precision, efficiency_info, input_shape):
    """新增论文制表用 TXT，不改变原有预测和精度输出。"""
    output_path = os.path.join(output_dir, "验证集精度与效率指标.txt")
    pv_index = 1
    precision = float(Precision[pv_index])
    recall = float(PA_Recall[pv_index])
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = float(IoUs[pv_index])
    miou = float(np.nanmean(IoUs))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SegFormer 语义分割论文制表指标\n")
        f.write("=" * 96 + "\n\n")

        f.write("一、精度指标\n")
        f.write(
            "{:<14s}{:<10s}{:>12s}{:>12s}{:>14s}{:>12s}{:>12s}\n".format(
                "Model", "Bands", "F1", "IoU", "Precision", "Recall", "mIoU"
            )
        )
        f.write("-" * 86 + "\n")
        f.write(
            "{:<14s}{:<10s}{:>12.3f}{:>12.3f}{:>14.3f}{:>12.3f}{:>12.3f}\n\n".format(
                MODEL_NAME,
                paper_band_label(band_mode),
                f1,
                iou,
                precision,
                recall,
                miou,
            )
        )

        f.write("二、效率指标\n")
        f.write(
            "{:<14s}{:>14s}{:>14s}{:>24s}{:>14s}\n".format(
                "Model", "Params(M)", "FLOPs(G)", "Latency(ms/image)", "FPS"
            )
        )
        f.write("-" * 80 + "\n")
        f.write(
            "{:<14s}{:>14.3f}{:>14.3f}{:>24.3f}{:>14.3f}\n\n".format(
                MODEL_NAME,
                efficiency_info["params_m"],
                efficiency_info["flops_g"],
                efficiency_info["latency_ms_per_image"],
                efficiency_info["fps"],
            )
        )

        f.write("评测口径：\n")
        f.write("1. 精度指标来自当前验证集；Precision、Recall、F1和IoU均为光伏板前景类指标。\n")
        f.write("2. F1按照 2×Precision×Recall/(Precision+Recall) 计算，mIoU为背景与光伏板IoU的平均值。\n")
        f.write(
            "3. 效率输入为 batch=1、{}×{}×{}，FP32，不使用TTA，不包含数据读取、DataLoader、指标计算和后处理。\n".format(
                in_channels,
                int(input_shape[0]),
                int(input_shape[1]),
            )
        )
        f.write("4. FLOPs由PyTorch Profiler统计支持的算子；一次乘法和一次加法合计为2 FLOPs。\n")
        f.write(
            "5. 延迟/FPS：预热{}次，连续前向测试{}次，延迟取平均值，FPS=1000/延迟(ms)。\n".format(
                efficiency_info["warmup_iters"],
                efficiency_info["benchmark_iters"],
            )
        )
        f.write("6. 设备：{}。\n".format(efficiency_info["cuda_device_name"] or efficiency_info["device"]))

    return output_path


"""
多光谱 SegFormer mIoU 评估脚本。

这个脚本专门用于当前光伏板二分类多光谱实验，和原始 get_miou.py 分开，
避免破坏原来的 RGB/VOC 示例流程。

数据目录仍然使用 VOC 风格：
VOCdevkit/VOC2007/JPEGImages/xxx.tif
VOCdevkit/VOC2007/SegmentationClass/xxx.png
VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt

注意：
1. 输入影像后缀和波段选择来自 multispectral_config.py：
   - image_ext = ".tif"
   - selected_bands = [1, 2, 3, 4, 5, 6]
2. 标签 png 必须是类别索引图：
   - background = 0
   - photovoltaic = 1
3. 生成的预测 png 也是单通道类别索引图，不是彩色可视化图。
"""


if __name__ == "__main__":
    # miou_mode:
    # 0 = 先生成预测结果，再计算 mIoU
    # 1 = 只生成预测结果
    # 2 = 只计算 mIoU，适合已经生成过预测结果时使用
    miou_mode = 0

    # 光伏板二分类：背景 + 光伏板。
    num_classes = 2
    name_classes = ["background", "photovoltaic"]

    VOCdevkit_path = "VOCdevkit"

    image_set_path = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt")
    image_ids = open(image_set_path, "r").read().splitlines()

    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "精度指标结果miou_out"
    pred_dir = os.path.join(miou_out_path, "detection-results")
    segformer = None

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        segformer = SegFormer_Segmentation()
        print("Load model done.")
        print("image_ext:", image_ext)
        print("selected_bands:", selected_bands)

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            name = image_id.split()[0]
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages", name + image_ext)
            image = segformer.open_image(image_path)
            image = segformer.get_miou_png(image)
            image.save(os.path.join(pred_dir, name + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir,
            pred_dir,
            [image_id.split()[0] for image_id in image_ids],
            num_classes,
            name_classes,
        )
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        # miou_mode=2 时前面不会加载模型；为生成效率指标，在这里单独加载。
        if segformer is None:
            print("Load model for efficiency benchmark.")
            segformer = SegFormer_Segmentation()
            print("Load model done.")

        # 单模型效率测试不计 DataParallel 的额外调度开销，与其他模型保持一致。
        benchmark_model = (
            segformer.net.module
            if isinstance(segformer.net, torch.nn.DataParallel)
            else segformer.net
        )
        benchmark_device = next(benchmark_model.parameters()).device
        print(
            "Benchmarking model efficiency: batch=1, warmup={}, iterations={}...".format(
                EFFICIENCY_WARMUP_ITERS,
                EFFICIENCY_BENCHMARK_ITERS,
            )
        )
        efficiency_info = benchmark_model_efficiency(
            benchmark_model,
            segformer.input_shape,
            benchmark_device,
            EFFICIENCY_WARMUP_ITERS,
            EFFICIENCY_BENCHMARK_ITERS,
        )
        summary_path = save_accuracy_efficiency_txt(
            miou_out_path,
            IoUs,
            PA_Recall,
            Precision,
            efficiency_info,
            segformer.input_shape,
        )
        print(
            "Efficiency: Params={:.3f}M FLOPs={:.3f}G Latency={:.3f}ms/image FPS={:.3f}".format(
                efficiency_info["params_m"],
                efficiency_info["flops_g"],
                efficiency_info["latency_ms_per_image"],
                efficiency_info["fps"],
            )
        )
        print("Saved validation accuracy and efficiency summary to:", summary_path)
