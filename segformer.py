import colorsys
import copy
import time

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from multispectral_config import (image_ext, in_channels, selected_bands,
                                  trained_model_path, vis_bands)
from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class SegFormer_Segmentation(object):
    _defaults = {
        # 推理默认使用训练完成后的权重。训练结果会按 band_mode 保存到
        # logs/rgb、logs/4band 或 logs/6band 下。
        "model_path": trained_model_path,

        # 光伏板二分类：background + photovoltaic panel。
        "num_classes": 2,

        # SegFormer 主干规模：b0/b1/b2/b3/b4/b5。
        "phi": "b0",

        # 需要与训练时 input_shape 保持一致。
        "input_shape": [256, 256],

        # 可视化方式：
        # 0 = 原图与分割结果混合；1 = 仅分割结果；2 = 仅保留目标区域。
        "mix_type": 0,

        # 多光谱配置，与 train.py 使用同一份 multispectral_config.py。
        "image_ext": image_ext,
        "selected_bands": selected_bands,
        "in_channels": in_channels,
        "vis_bands": vis_bands,

        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                (0, 64, 128), (128, 64, 12)
            ]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        show_config(**{key: getattr(self, key) for key in self._defaults.keys()})

    def generate(self, onnx=False):
        self.net = SegFormer(
            num_classes=self.num_classes,
            phi=self.phi,
            pretrained=False,
            in_channels=self.in_channels,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(self.model_path, map_location=device)
        model_dict = self.net.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        for key, value in pretrained_dict.items():
            if key in model_dict.keys() and np.shape(model_dict[key]) == np.shape(value):
                temp_dict[key] = value
                load_key.append(key)
            else:
                no_load_key.append(key)
        model_dict.update(temp_dict)
        self.net.load_state_dict(model_dict)
        self.net = self.net.eval()

        print("{} model, and classes loaded.".format(self.model_path))
        print("Successful Load Key Num: {}".format(len(load_key)))
        print("Fail To Load Key Num: {}".format(len(no_load_key)))

        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def read_tif(self, image_path):
        """
        Read a multispectral tif and select the same bands used for training.

        rasterio returns (C, H, W). The rest of this project uses HWC before
        converting to PyTorch BCHW tensors, so we transpose here.
        """
        with rasterio.open(image_path) as src:
            if self.selected_bands is None:
                band_indexes = list(range(1, src.count + 1))
            else:
                band_indexes = self.selected_bands
            image = src.read(indexes=band_indexes)
            image = np.transpose(image, (1, 2, 0))
        return image

    def open_image(self, image_path):
        """
        Open an input image according to the configured image type.

        For tif/tiff, return a HWC numpy array with selected bands.
        For jpg/png, return a PIL image and keep the original RGB flow.
        """
        if self.image_ext.lower() in [".tif", ".tiff"] or image_path.lower().endswith((".tif", ".tiff")):
            return self.read_tif(image_path)
        return Image.open(image_path)

    def resize_multiband_image(self, image, size):
        """
        Letterbox resize for multispectral images.

        PIL's RGB canvas would drop extra channels, so 4/6-band images use a
        numpy canvas and keep every selected band.
        """
        w, h = size
        ih, iw = image.shape[:2]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        channels = image.shape[2]
        new_image = np.full((h, w, channels), 128, dtype=image.dtype)
        new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
        return new_image, nw, nh

    def multiband_to_rgb(self, image):
        """
        Build a visible RGB preview from multispectral input.

        The model still uses all selected channels. This method is only for
        displaying blended prediction results in mix_type 0/2.
        """
        channel_indexes = []
        for band in self.vis_bands:
            if self.selected_bands is not None and band in self.selected_bands:
                channel_indexes.append(self.selected_bands.index(band))

        if len(channel_indexes) != 3:
            channel_indexes = list(range(min(3, image.shape[2])))

        rgb = image[:, :, channel_indexes].astype(np.float32)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)

        out = np.zeros_like(rgb[:, :, :3], dtype=np.uint8)
        for i in range(3):
            band = rgb[:, :, i]
            low, high = np.percentile(band, (2, 98))
            if high <= low:
                high = low + 1.0
            out[:, :, i] = np.clip((band - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def prepare_input(self, image):
        """
        Convert PIL RGB or multispectral HWC numpy image into model input.

        Returns image_data, resized valid width/height, original width/height,
        and an RGB PIL preview image for visualization.
        """
        is_multispectral = not isinstance(image, Image.Image)

        if is_multispectral:
            old_img = self.multiband_to_rgb(image)
            original_h, original_w = image.shape[:2]
            image_data, nw, nh = self.resize_multiband_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.expand_dims(
                np.transpose(preprocess_input(np.array(image_data, np.float32), is_multispectral=True), (2, 0, 1)),
                0,
            )
        else:
            image = cvtColor(image)
            old_img = copy.deepcopy(image)
            original_h = np.array(image).shape[0]
            original_w = np.array(image).shape[1]
            image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.expand_dims(
                np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
                0,
            )

        return image_data, nw, nh, original_w, original_h, old_img

    def predict_mask(self, image):
        image_data, nw, nh, original_w, original_h, old_img = self.prepare_input(image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return pr, original_w, original_h, old_img

    def detect_image(self, image, count=False, name_classes=None):
        pr, original_w, original_h, old_img = self.predict_mask(image)

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = original_h * original_w
            print("-" * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print("-" * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    class_name = name_classes[i] if name_classes is not None else str(i)
                    print("|%25s | %15s | %14.2f%%|" % (class_name, str(num), ratio))
                    print("-" * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [original_h, original_w, -1],
            )
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)
        elif self.mix_type == 1:
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [original_h, original_w, -1],
            )
            image = Image.fromarray(np.uint8(seg_img))
        else:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype("uint8")
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        image_data, nw, nh, _, _, _ = self.prepare_input(image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                        int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        return (t2 - t1) / test_interval

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, self.in_channels, *self.input_shape).to("cpu")
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        print(f"Starting export with onnx {onnx.__version__}.")
        torch.onnx.export(
            self.net,
            im,
            f=model_path,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=input_layer_names,
            output_names=output_layer_names,
            dynamic_axes=None,
        )

        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        if simplify:
            import onnxsim
            print(f"Simplifying with onnx-simplifier {onnxsim.__version__}.")
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None,
            )
            assert check, "assert check failed"
            onnx.save(model_onnx, model_path)

        print("Onnx model save as {}".format(model_path))

    def get_miou_png(self, image):
        pr, _, _, _ = self.predict_mask(image)
        return Image.fromarray(np.uint8(pr))
