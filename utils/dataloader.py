import os

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from multispectral_config import normalization_config
from utils.utils import cvtColor, preprocess_input


class SegmentationDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        train,
        dataset_path,
        image_ext=".jpg",
        selected_bands=None,
        augmentation_config=None,
    ):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

        # image_ext 控制输入影像类型：
        # - ".jpg" / ".png"：保持原来的普通 RGB 语义分割流程；
        # - ".tif" / ".tiff"：使用 rasterio 读取多光谱影像。
        self.image_ext          = image_ext

        # selected_bands 只在 tif 多光谱输入时使用。
        # rasterio 的波段编号从 1 开始，例如 6 波段 Sentinel-2 tif 若顺序为
        # [B2, B3, B4, B8, B11, B12]，则：
        # - [3, 2, 1] 表示真彩色 RGB 对比实验 [B4, B3, B2]；
        # - [1, 2, 3, 4, 5, 6] 表示完整六波段实验。
        self.selected_bands     = selected_bands
        self.augmentation_config = augmentation_config or {"enabled": False}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + self.image_ext)
        label_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")

        if self.image_ext.lower() in [".tif", ".tiff"]:
            image = self.read_tif(image_path)
        else:
            image = Image.open(image_path)
        label = Image.open(label_path)

        is_multispectral = self.image_ext.lower() in [".tif", ".tiff"]
        if self._use_training_augmentation(image):
            image, label = self.get_multispectral_training_data(image, label, self.input_shape)
        else:
            image, label = self.get_random_data(image, label, self.input_shape, random=self.train)

        # 统一转换成 HWC numpy，再交给 preprocess_input。
        # 现在这一阶段先保证多光谱维度能进训练管线；下一步会改 preprocess_input，
        # 让 tif 多光谱走 /10000、clip、mean/std，RGB 仍保持原来的 /255。
        image = np.array(image, np.float64)
        if len(np.shape(image)) == 2:
            image = np.expand_dims(image, -1)

        image = np.transpose(preprocess_input(image, is_multispectral=is_multispectral), [2, 0, 1])
        label = np.array(label)
        label[label >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape(
            (int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1)
        )

        return image, label, seg_labels

    def read_tif(self, image_path):
        """
        读取多光谱 tif 文件。

        rasterio 读出的数组是 (C, H, W)，而本仓库原来的增强和预处理逻辑使用
        (H, W, C)，因此这里会转成 HWC。selected_bands 为 None 时读取全部波段。
        """
        with rasterio.open(image_path) as src:
            if self.selected_bands is None:
                band_indexes = list(range(1, src.count + 1))
            else:
                band_indexes = self.selected_bands

            image = src.read(indexes=band_indexes)
            image = np.transpose(image, (1, 2, 0))
        return image

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def _use_training_augmentation(self, image):
        return (
            self.train
            and self.augmentation_config.get("enabled", False)
            and self.image_ext.lower() in [".tif", ".tiff"]
            and isinstance(image, np.ndarray)
            and image.ndim == 3
        )

    def _to_reflectance(self, image):
        scale = float(normalization_config.get("reflectance_scale", 1.0))
        if scale <= 0:
            raise ValueError("normalization_config['reflectance_scale'] must be greater than 0.")
        return image.astype(np.float32, copy=True) / scale

    def _from_reflectance(self, image):
        scale = float(normalization_config.get("reflectance_scale", 1.0))
        return (image.astype(np.float32, copy=False) * scale).astype(np.float32)

    def get_multispectral_training_data(self, image, label, input_shape):
        target = np.array(label, dtype=np.uint8)
        reflectance = self._to_reflectance(image)
        reflectance, target = self._augment_training_sample(reflectance, target)

        h, w = input_shape
        image = self._from_reflectance(reflectance)
        image, nw, nh = self.resize_image(image, (w, h))

        label = Image.fromarray(target).resize((nw, nh), Image.NEAREST)
        new_label = Image.new("L", [w, h], 0)
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
        return image, new_label

    def _augment_training_sample(self, image, target):
        cfg = self.augmentation_config

        if self.rand() < cfg.get("geometry_prob", 0.0):
            image, target = self._augment_geometry(image, target)

        if self.rand() < cfg.get("scale_prob", 0.0):
            image, target = self._augment_random_scale(image, target)

        if self.rand() < cfg.get("reflectance_prob", 0.0):
            image = self._augment_reflectance(image)

        if self.rand() < cfg.get("shadow_prob", 0.0):
            image = self._augment_shadow(image)

        if self.rand() < cfg.get("noise_prob", 0.0):
            image = self._augment_noise(image)

        return image.astype(np.float32), target.astype(np.uint8)

    def _augment_geometry(self, image, target):
        op = np.random.choice(["hflip", "vflip", "rot90", "rot180", "rot270"])
        if op == "hflip":
            return np.ascontiguousarray(image[:, ::-1, :]), np.ascontiguousarray(target[:, ::-1])
        if op == "vflip":
            return np.ascontiguousarray(image[::-1, :, :]), np.ascontiguousarray(target[::-1, :])
        k = {"rot90": 1, "rot180": 2, "rot270": 3}[op]
        return np.rot90(image, k=k).copy(), np.rot90(target, k=k).copy()

    def _augment_reflectance(self, image):
        cfg = self.augmentation_config
        global_low, global_high = cfg.get("reflectance_global_range", [0.90, 1.10])
        band_low, band_high = cfg.get("reflectance_band_range", [0.95, 1.05])
        global_factor = self.rand(global_low, global_high)
        band_factors = np.random.uniform(
            band_low,
            band_high,
            size=(1, 1, image.shape[2]),
        ).astype(np.float32)
        return image * global_factor * band_factors

    def _augment_shadow(self, image):
        cfg = self.augmentation_config
        factor_low, factor_high = cfg.get("shadow_factor_range", [0.75, 0.90])
        radius_low, radius_high = cfg.get("shadow_radius_range", [0.25, 0.45])
        h, w = image.shape[:2]
        center_y = self.rand(-0.5, 0.5)
        center_x = self.rand(-0.5, 0.5)
        radius = self.rand(radius_low, radius_high)
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        shadow = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(radius, 1e-6))
        factor = self.rand(factor_low, factor_high)
        shadow_map = 1.0 - (1.0 - factor) * shadow
        return image * shadow_map[:, :, None]

    def _augment_noise(self, image):
        sigma_low, sigma_high = self.augmentation_config.get("noise_sigma_range", [0.003, 0.008])
        sigma = self.rand(sigma_low, sigma_high)
        noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        return image + noise

    def _augment_random_scale(self, image, target):
        crop_low, crop_high = self.augmentation_config.get("scale_crop_range", [0.85, 1.00])
        ratio = self.rand(crop_low, crop_high)
        h, w = image.shape[:2]
        crop_h = max(8, int(h * ratio))
        crop_w = max(8, int(w * ratio))
        top = np.random.randint(0, max(0, h - crop_h) + 1)
        left = np.random.randint(0, max(0, w - crop_w) + 1)

        image_crop = image[top:top + crop_h, left:left + crop_w, :]
        target_crop = target[top:top + crop_h, left:left + crop_w]
        image = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        if image.ndim == 2:
            image = image[:, :, None]
        return image.astype(np.float32), target.astype(np.uint8)

    def get_image_size(self, image):
        if isinstance(image, Image.Image):
            iw, ih = image.size
        else:
            ih, iw = image.shape[:2]
        return iw, ih

    def resize_image(self, image, size):
        """
        按原仓库 letterbox 方式 resize。

        PIL RGB 图像继续使用 Image.resize；多光谱 numpy 数组不能用 Image.new('RGB')，
        所以用 np.full 创建多通道背景，再把 resize 后的影像贴进去。
        """
        w, h        = size
        iw, ih      = self.get_image_size(image)
        scale       = min(w / iw, h / ih)
        nw          = int(iw * scale)
        nh          = int(ih * scale)

        if isinstance(image, Image.Image):
            image       = image.resize((nw, nh), Image.BICUBIC)
            new_image   = Image.new("RGB", [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            image       = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            channels    = image.shape[2]
            new_image   = np.full((h, w, channels), 128, dtype=image.dtype)
            new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image

        return new_image, nw, nh

    def paste_multiband(self, canvas, image, dx, dy):
        """
        Paste a multiband numpy image onto a fixed-size canvas with cropping.

        PIL.Image.paste supports negative offsets and automatically crops the
        part outside the canvas. Numpy assignment does not, so random scale
        augmentation can fail when the resized image is larger than input_shape.
        This helper reproduces the safe paste behavior for 4/6-band arrays.
        """
        canvas_h, canvas_w = canvas.shape[:2]
        image_h, image_w = image.shape[:2]

        dst_x1 = max(dx, 0)
        dst_y1 = max(dy, 0)
        dst_x2 = min(dx + image_w, canvas_w)
        dst_y2 = min(dy + image_h, canvas_h)

        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            return canvas

        src_x1 = dst_x1 - dx
        src_y1 = dst_y1 - dy
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = image[src_y1:src_y2, src_x1:src_x2, :]
        return canvas

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        is_multiband = not isinstance(image, Image.Image)

        # 只有普通 PIL 图像才转 RGB。多光谱 tif 已经是 HWC numpy 数组，
        # 强制转 RGB 会丢掉 NIR/SWIR 等波段。
        if not is_multiband:
            image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = self.get_image_size(image)
        h, w   = input_shape

        if not random:
            image, nw, nh = self.resize_image(image, (w, h))

            label       = label.resize((nw, nh), Image.NEAREST)
            new_label   = Image.new("L", [w, h], 0)
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return image, new_label

        # 随机缩放与长宽扰动。RGB 和多光谱共用同一套几何增强，
        # 这样不同波段实验之间的数据增强强度保持一致，便于做公平对比。
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale  = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        if is_multiband:
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # 随机水平翻转。多光谱 numpy 数组直接按宽度维翻转。
        flip = self.rand() < 0.5
        if flip:
            if is_multiband:
                image = image[:, ::-1, :]
            else:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机放置到固定输入尺寸画布中，保持原仓库 letterbox 风格。
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        if is_multiband:
            channels = image.shape[2]
            new_image = np.full((h, w, channels), 128, dtype=image.dtype)
            new_image = self.paste_multiband(new_image, image, dx, dy)
        else:
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))

        new_label = Image.new("L", (w, h), 0)
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image)

        # 高斯模糊属于空间增强，多光谱也可以使用；它会对每个通道独立处理。
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # 小角度旋转属于空间增强，RGB 和多光谱都可以使用。
        rotate = self.rand() < 0.25
        if rotate:
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            # OpenCV 对 5 通道及以上图像不支持 INTER_CUBIC warpAffine。
            # 多光谱影像这里使用 INTER_LINEAR，RGB 仍保留原来的 INTER_CUBIC。
            interpolation = cv2.INTER_LINEAR if is_multiband else cv2.INTER_CUBIC
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=interpolation, borderValue=128)
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        if is_multiband:
            # 多光谱波段不是 RGB 颜色空间，不能做 HSV 色彩增强。
            # 这里直接返回几何增强后的多通道数组。
            return image_data, label

        image_data = image_data.astype(np.uint8)

        # RGB 图像保留原仓库的 HSV 色彩扰动。
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def seg_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
