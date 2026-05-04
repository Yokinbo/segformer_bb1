import os

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

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

        image, label = self.get_random_data(image, label, self.input_shape, random=self.train)

        # 统一转换成 HWC numpy，再交给 preprocess_input。
        # 现在这一阶段先保证多光谱维度能进训练管线；下一步会改 preprocess_input，
        # 让 tif 多光谱走 /10000、clip、mean/std，RGB 仍保持原来的 /255。
        image = np.array(image, np.float64)
        if len(np.shape(image)) == 2:
            image = np.expand_dims(image, -1)

        is_multispectral = self.image_ext.lower() in [".tif", ".tiff"]
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
