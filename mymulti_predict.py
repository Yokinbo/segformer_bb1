import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from multispectral_config import (
    image_ext,
    in_channels,
    selected_bands,
    trained_model_path,
    vis_bands,
)
from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image

try:
    import rasterio
except ImportError:
    rasterio = None


# ===================== 可编辑配置区 =====================
# 直接运行:
#   python mymulti_predict.py
#
# INPUT_PATH 支持:
#   1. 单张图片，例如 r"论文制图\测试图\原图\shenmu39.tif"
#   2. 图片文件夹，例如 r"论文制图\测试图\原图"
EDITABLE_CONFIG = {
    "input_path": r"论文制图\测试图\原图",
    "label_dir": r"论文制图\测试图\label标签",
    "weights": trained_model_path,
    "output_dir": r"论文制图\6band测试结果",
    "device": "cuda:0",
    "input_size": 256,
    "num_classes": 2,
    "phi": "b0",
    "target_class": 1,
    "threshold": 0.5,
    "suffixes": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    "label_suffixes": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
    "save_mask": True,
    "save_overlay": True,
    "save_prob": False,
    "save_confusion": True,
}
# =======================================================


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def imwrite_unicode(path, image):
    """OpenCV on Windows may fail on Chinese paths; imencode+tofile is safer."""
    path = Path(path)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"failed to encode image for: {path}")
    encoded.tofile(str(path))


def read_tif_bands(image_path, bands):
    if rasterio is None:
        raise ImportError("rasterio is required to read tif images. Install it with `pip install rasterio`.")

    with rasterio.open(image_path) as src:
        invalid_bands = [band for band in bands if band < 1 or band > src.count]
        if invalid_bands:
            raise ValueError(f"{image_path} has {src.count} bands, invalid bands: {invalid_bands}")
        image = src.read(indexes=bands)
    return np.transpose(image, (1, 2, 0))


def read_predict_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        image = read_tif_bands(image_path, selected_bands)
        if image.shape[2] != in_channels:
            raise ValueError(f"Read {image.shape[2]} channels from {image_path}, model expects {in_channels}.")
        return image

    if in_channels != 3:
        raise ValueError("jpg/png prediction only works when in_channels=3.")
    image = Image.open(image_path)
    image = cvtColor(image)
    return np.array(image)


def make_preview_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        rgb = read_tif_bands(image_path, vis_bands).astype(np.float32)
        out = np.zeros_like(rgb[:, :, :3], dtype=np.uint8)
        for channel in range(3):
            band = rgb[:, :, channel]
            low, high = np.percentile(band, (2, 98))
            if high <= low:
                out[:, :, channel] = 0
            else:
                out[:, :, channel] = np.clip((band - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        return out

    image = Image.open(image_path)
    image = cvtColor(image)
    return np.array(image)


def resize_multiband_image(image, size):
    w, h = size
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, image.shape[2]), 128, dtype=image.dtype)
    top = (h - nh) // 2
    left = (w - nw) // 2
    canvas[top:top + nh, left:left + nw, :] = image
    return canvas, nw, nh


def prepare_tensor(image, input_size, device, is_multispectral):
    if is_multispectral:
        resized, nw, nh = resize_multiband_image(image, (input_size, input_size))
        image_data = preprocess_input(np.array(resized, np.float32), is_multispectral=True)
    else:
        pil_image = Image.fromarray(np.uint8(image))
        resized, nw, nh = resize_image(pil_image, (input_size, input_size))
        image_data = preprocess_input(np.array(resized, np.float32), is_multispectral=False)

    image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
    return torch.from_numpy(image_data).float().to(device), nw, nh


def load_model(args, device):
    model = SegFormer(
        num_classes=args.num_classes,
        phi=args.phi,
        pretrained=False,
        in_channels=in_channels,
    )
    weights = torch.load(args.weights, map_location="cpu")
    if "model" in weights:
        weights = weights["model"]
    weights = {key.replace("module.", "", 1): value for key, value in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    return model


def collect_input_images(input_path, suffixes):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        suffix_set = {suffix.lower() for suffix in suffixes}
        return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in suffix_set])
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def find_label_for_image(image_path, label_dir, label_suffixes):
    if not label_dir:
        return None

    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir does not exist: {label_dir}")

    for suffix in label_suffixes:
        label_path = label_dir / f"{image_path.stem}{suffix}"
        if label_path.exists():
            return label_path
    return None


def read_label_mask(label_path, target_shape):
    label = np.array(Image.open(label_path).convert("L"))
    if label.shape != target_shape:
        label = cv2.resize(label, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (label > 0).astype(np.uint8)


def save_prediction_outputs(prob, preview_img, threshold, output_mask, output_overlay):
    pred_mask = (prob > threshold).astype(np.uint8)
    if output_mask:
        imwrite_unicode(output_mask, pred_mask * 255)

    if output_overlay:
        overlay = preview_img.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255
        overlay = np.where(pred_mask[..., None] > 0, (0.55 * overlay + 0.45 * red), overlay)
        imwrite_unicode(output_overlay, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return pred_mask


def save_probability(prob, output_prob):
    imwrite_unicode(output_prob, np.clip(prob * 255.0, 0, 255).astype(np.uint8))


def save_viewable_label(label_mask, output_label):
    imwrite_unicode(output_label, label_mask.astype(np.uint8) * 255)


def save_confusion_map(pred_mask, label_mask, output_confusion):
    # TN=black, FP=blue, TP=white, FN=red.
    pred01 = (pred_mask > 0).astype(np.uint8)
    label01 = (label_mask > 0).astype(np.uint8)

    rgb = np.zeros((label01.shape[0], label01.shape[1], 3), dtype=np.uint8)
    tp = (pred01 == 1) & (label01 == 1)
    fp = (pred01 == 1) & (label01 == 0)
    fn = (pred01 == 0) & (label01 == 1)

    rgb[tp] = [255, 255, 255]
    rgb[fp] = [0, 0, 255]
    rgb[fn] = [255, 0, 0]
    imwrite_unicode(output_confusion, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def predict_one_image(model, image_path, args, device, warmup_done):
    input_image = read_predict_image(str(image_path))
    preview_img = make_preview_image(str(image_path))
    original_h, original_w = input_image.shape[:2]
    is_multispectral = image_path.suffix.lower() in [".tif", ".tiff"]
    img_tensor, nw, nh = prepare_tensor(input_image, args.input_size, device, is_multispectral)

    with torch.no_grad():
        if not warmup_done[0]:
            init_img = torch.zeros((1, in_channels, args.input_size, args.input_size), device=device)
            model(init_img)
            warmup_done[0] = True

        t_start = time_synchronized()
        pred = model(img_tensor)[0]
        t_end = time_synchronized()

    pred = F.softmax(pred.permute(1, 2, 0), dim=-1).cpu().numpy()
    top = int((args.input_size - nh) // 2)
    left = int((args.input_size - nw) // 2)
    pred = pred[top:top + nh, left:left + nw]
    pred = cv2.resize(pred, dsize=(original_w, original_h), interpolation=cv2.INTER_LINEAR)

    if args.target_class >= pred.shape[-1]:
        raise ValueError(f"target_class={args.target_class} is out of range for {pred.shape[-1]} classes.")
    prob = pred[:, :, args.target_class]

    output_dir = Path(args.output_dir)
    mask_path = output_dir / "模型预测mask_0-255" / f"{image_path.stem}_mask.png" if args.save_mask else None
    overlay_path = output_dir / "红色半透明叠加图" / f"{image_path.stem}_overlay.png" if args.save_overlay else None
    prob_path = output_dir / "模型预测概率图" / f"{image_path.stem}_prob.png" if args.save_prob else None
    label_view_path = output_dir / "人工标签可视化0-255" / f"{image_path.stem}_label.png" if args.save_confusion else None
    confusion_path = output_dir / "TP_FP_FN_TN彩色误差图" / f"{image_path.stem}_confusion.png" if args.save_confusion else None

    for path in [mask_path, overlay_path, prob_path, label_view_path, confusion_path]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    pred_mask = save_prediction_outputs(prob, preview_img, args.threshold, mask_path, overlay_path)
    if prob_path is not None:
        save_probability(prob, prob_path)

    if confusion_path is not None:
        label_path = find_label_for_image(image_path, args.label_dir, args.label_suffixes)
        if label_path is None:
            print(f"[warn] no label found for {image_path.name}, skip confusion map")
        else:
            label_mask = read_label_mask(label_path, pred_mask.shape)
            save_viewable_label(label_mask, label_view_path)
            save_confusion_map(pred_mask, label_mask, confusion_path)

    print(f"[done] {image_path.name} inference={t_end - t_start:.4f}s")


def main(args):
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights file does not exist: {args.weights}")

    image_paths = collect_input_images(args.input_path, args.suffixes)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input_path}")

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    print("Current SegFormer multispectral prediction config:")
    print(f"  image_ext     : {image_ext}")
    print(f"  selected_bands: {selected_bands}")
    print(f"  in_channels   : {in_channels}")
    print(f"  phi           : {args.phi}")
    print(f"  num_classes   : {args.num_classes}")
    print(f"  target_class  : {args.target_class}")
    print(f"  weights       : {args.weights}")
    print(f"  input_path    : {args.input_path}")
    print(f"  label_dir     : {args.label_dir or '(disabled)'}")
    print(f"  image_count   : {len(image_paths)}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  device        : {device}")

    model = load_model(args, device)
    warmup_done = [False]
    for image_path in image_paths:
        predict_one_image(model, image_path, args, device, warmup_done)

    print("Saved outputs to:", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer multispectral prediction")
    parser.add_argument("--input-path", default=EDITABLE_CONFIG["input_path"], help="input image file or folder")
    parser.add_argument("--label-dir", default=EDITABLE_CONFIG["label_dir"], help="manual label folder")
    parser.add_argument("--weights", default=EDITABLE_CONFIG["weights"], help="model weights path")
    parser.add_argument("--output-dir", default=EDITABLE_CONFIG["output_dir"], help="output folder")
    parser.add_argument("--device", default=EDITABLE_CONFIG["device"], help="prediction device")
    parser.add_argument("--input-size", default=EDITABLE_CONFIG["input_size"], type=int, help="square inference size")
    parser.add_argument("--num-classes", default=EDITABLE_CONFIG["num_classes"], type=int, help="class count")
    parser.add_argument("--phi", default=EDITABLE_CONFIG["phi"], help="SegFormer backbone: b0/b1/b2/b3/b4/b5")
    parser.add_argument("--target-class", default=EDITABLE_CONFIG["target_class"], type=int, help="foreground class id")
    parser.add_argument("--threshold", default=EDITABLE_CONFIG["threshold"], type=float, help="binary mask threshold")
    parser.add_argument("--suffixes", nargs="+", default=EDITABLE_CONFIG["suffixes"], help="image suffixes")
    parser.add_argument("--label-suffixes", nargs="+", default=EDITABLE_CONFIG["label_suffixes"], help="label suffixes")
    parser.add_argument("--save-mask", action="store_true", default=EDITABLE_CONFIG["save_mask"])
    parser.add_argument("--no-save-mask", action="store_false", dest="save_mask")
    parser.add_argument("--save-overlay", action="store_true", default=EDITABLE_CONFIG["save_overlay"])
    parser.add_argument("--no-save-overlay", action="store_false", dest="save_overlay")
    parser.add_argument("--save-prob", action="store_true", default=EDITABLE_CONFIG["save_prob"])
    parser.add_argument("--no-save-prob", action="store_false", dest="save_prob")
    parser.add_argument("--save-confusion", action="store_true", default=EDITABLE_CONFIG["save_confusion"])
    parser.add_argument("--no-save-confusion", action="store_false", dest="save_confusion")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
