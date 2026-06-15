"""
多光谱训练统一配置文件。

这个文件先只负责“集中保存实验配置”，后续会在 train.py、segformer.py、
utils/dataloader.py 和 utils/utils.py 中逐步接入这些配置。

使用建议：
1. 做 Sentinel-2 光伏板多光谱分割实验时，优先修改 band_mode。
2. 做普通 RGB 对比实验时，可以把 band_mode 改成 "rgb"。
3. 所有训练、验证、推理脚本都应尽量从这里读取波段、输入通道数和标准化参数，
   避免不同脚本之间配置不一致。
"""

# -------------------------------------------------------------------------
# 模型权重路径
# -------------------------------------------------------------------------
# 训练完成后的权重路径，后续主要给 predict.py / get_miou.py / segformer.py 推理使用。
# 第一阶段只是放一个默认占位路径；真正训练完成后，可以改成 logs 下的 best_epoch_weights.pth。
trained_model_path = "logs/6band/best_epoch_weights.pth"

# -------------------------------------------------------------------------
# 输入影像格式
# -------------------------------------------------------------------------
# 当前多光谱实验默认使用 .tif。
# 如果后续要做普通 jpg RGB 对比实验，可以改成 ".jpg"。
image_ext = ".tif"


# -------------------------------------------------------------------------
# 波段模式
# -------------------------------------------------------------------------
# Sentinel-2 常用波段说明：
# - B2  : Blue，蓝光
# - B3  : Green，绿光
# - B4  : Red，红光
# - B8  : NIR，近红外
# - B11 : SWIR1，短波红外 1
# - B12 : SWIR2，短波红外 2
#
# 可选模式：
# - "rgb"   : 使用真彩色三波段 [B4, B3, B2]
# - "4band" : 使用 [B2, B3, B4, B8]
# - "6band" : 使用 [B2, B3, B4, B8, B11, B12]
#
# 注意：
# 这里的波段编号使用 1-based 编号，也就是 rasterio 读取 tif 时的波段编号习惯。
# 假设你的 tif 内部波段顺序固定为：
# [1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
band_mode = "6band"

band_options = {
    # 真彩色 RGB。因为原始 tif 顺序是 [B2, B3, B4, B8, B11, B12]，
    # 所以真彩色对应 [B4, B3, B2] = [3, 2, 1]。
    "rgb": [3, 2, 1],

    # 四波段实验：可见光 + 近红外。
    "4band": [1, 2, 3, 4],

    # 六波段实验：可见光 + 近红外 + 短波红外。
    "6band": [1, 2, 3, 4, 5, 6],
}

# 当前实验实际读取的波段列表。
selected_bands = band_options[band_mode]

# 当前模型输入通道数。后续会传给 SegFormer 的第一层 patch embedding。
in_channels = len(selected_bands)

# 多光谱推理可视化时，仍然使用真彩色 [B4, B3, B2] 作为底图。
# 这只影响结果叠加显示，不影响模型训练输入。
vis_bands = [3, 2, 1]


# -------------------------------------------------------------------------
# 多光谱标准化配置
# -------------------------------------------------------------------------
# 这里定义训练和推理时共用的预处理参数。
#
# 设计思路：
# 1. Sentinel-2 tif 通常存储的是整数反射率，需要先除以 10000。
# 2. clip_min / clip_max 用于裁剪异常亮/暗像元，让输入分布更稳定。
# 3. mean / std 用于按波段标准化，便于不同波段在相近尺度上训练。
#
# 重要：
# - clip_min、clip_max、mean、std 的长度必须等于当前模式的通道数。
# - 这些数值来自训练集统计时才最严谨。
# - 目前先沿用 UNet_b 中同款统计参数，保证两套模型对比时预处理一致。
normalization_configs = {
    "rgb": {
        # selected_bands = [3, 2, 1] = [B4, B3, B2]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.037500, 0.051400, 0.029200],
        "clip_max": [0.323400, 0.250800, 0.195000],
        "enable_mean_std": True,
        "mean": [0.183408, 0.143386, 0.101527],
        "std": [0.066324, 0.047525, 0.035681],
    },
    "4band": {
        # selected_bands = [1, 2, 3, 4] = [B2, B3, B4, B8]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.029200, 0.051400, 0.037500, 0.094900],
        "clip_max": [0.195000, 0.250800, 0.323400, 0.408400],
        "enable_mean_std": True,
        "mean": [0.101527, 0.143386, 0.183408, 0.259956],
        "std": [0.035681, 0.047525, 0.066324, 0.070516],
    },
    "6band": {
        # selected_bands = [1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.029200, 0.051400, 0.037500, 0.094900, 0.146000, 0.082400],
        "clip_max": [0.195000, 0.250800, 0.323400, 0.408400, 0.469300, 0.455800],
        "enable_mean_std": True,
        "mean": [0.101527, 0.143386, 0.183408, 0.259956, 0.341008, 0.294076],
        "std": [0.035681, 0.047525, 0.066324, 0.070516, 0.071065, 0.079996],
    },
}

# -------------------------------------------------------------------------
# Training-time online data augmentation
# -------------------------------------------------------------------------
# Keep this block aligned with the U2Net multispectral experiments so model
# comparisons use the same augmentation policy. These augmentations are used
# only for the training split; validation and test samples stay unchanged.
train_augmentation_config = {
    "enabled": True,

    # Multispectral reflectance perturbation.
    "reflectance_prob": 0.50,
    "reflectance_global_range": [0.90, 1.10],
    "reflectance_band_range": [0.95, 1.05],

    # Geometry perturbation.
    "geometry_prob": 0.50,

    # Soft local shadow / thin cloud-shadow perturbation.
    "shadow_prob": 0.25,
    "shadow_factor_range": [0.75, 0.90],
    "shadow_radius_range": [0.25, 0.45],

    # Mild Gaussian noise in reflectance units.
    "noise_prob": 0.25,
    "noise_sigma_range": [0.003, 0.008],

    # Random scale by crop and resize back.
    "scale_prob": 0.20,
    "scale_crop_range": [0.85, 1.00],
}

# 当前波段模式对应的标准化配置。
normalization_config = normalization_configs[band_mode]
