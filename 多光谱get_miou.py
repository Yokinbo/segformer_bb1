import os

from tqdm import tqdm

from multispectral_config import image_ext, selected_bands
from segformer import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results


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
    miou_out_path = "miou_out_multispectral"
    pred_dir = os.path.join(miou_out_path, "detection-results")

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
