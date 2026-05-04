import random

import numpy as np
import torch
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _reshape_band_vector(values, channels, name):
    """
    将按波段配置的一维参数整理成 (1, 1, C)，方便和 HWC 影像广播运算。

    多光谱标准化要求每个通道都有自己的 clip / mean / std，因此这里会检查
    参数长度是否与当前输入通道数一致，避免 4 波段、6 波段配置混用。
    """
    arr = np.array(values, np.float32)
    if arr.ndim != 1 or arr.shape[0] != channels:
        raise ValueError(
            "{} 的长度应与输入通道数一致，当前通道数={}，配置长度={}。".format(
                name, channels, arr.shape[0] if arr.ndim == 1 else arr.shape
            )
        )
    return arr.reshape((1, 1, channels))

def preprocess_input(image, is_multispectral=False):
    """
    输入影像标准化。

    普通 RGB：
        沿用原 SegFormer 仓库的 ImageNet 标准化：
        (image - mean) / std，其中 image 是 0-255 的 RGB 数组。

    多光谱 tif：
        使用 multispectral_config.py 中的 normalization_config：
        1. 先除以 reflectance_scale，通常 Sentinel-2 为 /10000；
        2. 可选按波段 clip，压制异常亮/暗像元；
        3. 可选按波段 mean/std 标准化。

    为什么需要 is_multispectral：
        三波段 tif 的形状也是 HWC=3，单靠通道数无法判断它是普通 RGB jpg，
        还是 Sentinel-2 的 [B4, B3, B2]。因此 dataloader / 推理脚本应在读取
        tif 时显式传入 is_multispectral=True。
    """
    image = np.array(image, np.float32)

    if not is_multispectral:
        image -= np.array([123.675, 116.28, 103.53], np.float32)
        image /= np.array([58.395, 57.12, 57.375], np.float32)
        return image

    from multispectral_config import normalization_config

    channels = image.shape[2] if image.ndim == 3 else 1
    reflectance_scale = float(normalization_config.get("reflectance_scale", 10000.0))
    image = image / reflectance_scale

    if normalization_config.get("enable_clip", False):
        clip_min = _reshape_band_vector(normalization_config.get("clip_min"), channels, "clip_min")
        clip_max = _reshape_band_vector(normalization_config.get("clip_max"), channels, "clip_max")
        image = np.clip(image, clip_min, clip_max)

    if normalization_config.get("enable_mean_std", False):
        mean = _reshape_band_vector(normalization_config.get("mean"), channels, "mean")
        std = _reshape_band_vector(normalization_config.get("std"), channels, "std")
        image = (image - mean) / (std + 1e-8)

    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'b0' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b0_backbone_weights.pth",
        'b1' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b1_backbone_weights.pth",
        'b2' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b2_backbone_weights.pth",
        'b3' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b3_backbone_weights.pth",
        'b4' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b4_backbone_weights.pth",
        'b5' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b5_backbone_weights.pth",
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
