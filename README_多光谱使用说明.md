# SegFormer 多光谱光伏板语义分割使用说明

这个仓库已经从原来的 RGB SegFormer 语义分割，扩展为支持 Sentinel-2 多光谱 `.tif` 输入的版本。

当前默认任务是二分类语义分割：

```text
0 = background
1 = photovoltaic
```

## 1. 核心配置文件

多光谱实验的统一配置在：

```text
multispectral_config.py
```

常用配置项：

```python
image_ext = ".tif"
band_mode = "6band"
selected_bands = [1, 2, 3, 4, 5, 6]
in_channels = 6
vis_bands = [3, 2, 1]
```

默认假设你的 6 波段 tif 内部顺序是：

```text
[1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
```

可选波段模式：

```python
band_mode = "rgb"    # [B4, B3, B2], in_channels = 3
band_mode = "4band"  # [B2, B3, B4, B8], in_channels = 4
band_mode = "6band"  # [B2, B3, B4, B8, B11, B12], in_channels = 6
```

切换实验时，优先只改 `multispectral_config.py`，不要在多个脚本里手动改同一类参数。

## 2. 数据集目录

仍然使用 VOC 风格目录：

```text
VOCdevkit/
└── VOC2007/
    ├── JPEGImages/
    │   ├── 000001.tif
    │   ├── 000002.tif
    │   └── ...
    ├── SegmentationClass/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

`train.txt` 和 `val.txt` 中只写不带后缀的文件名：

```text
000001
000002
000003
```

标签 png 必须是类别索引图，不是彩色图：

```text
background = 0
photovoltaic = 1
```

如果你的标签里目标是 255，需要先转换成 1，否则训练和 mIoU 会不正确。

## 3. 模型输入通道

当前 SegFormer 已经支持动态输入通道：

```python
SegFormer(..., in_channels=in_channels)
```

RGB 原版：

```text
in_channels = 3
```

多光谱六波段：

```text
in_channels = 6
```

这个改动只影响第一层 patch embedding 的输入通道数，MiT encoder 和 MLP decoder 仍保持 SegFormer 结构。

论文中建议表述为：

```text
本文采用多光谱适配版 SegFormer-B0，将第一层 patch embedding 的输入通道数由 3 调整为 6，以适配 Sentinel-2 多光谱影像，其余网络结构保持不变。
```

## 4. 预训练权重

如果使用 RGB 主干预训练权重：

```python
pretrained = True
model_path = ""
```

当 `in_channels != 3` 时，第一层输入卷积的权重形状与 RGB 预训练权重不一致，会自动跳过；其余能匹配的 Transformer 主干权重会尽量加载。

如果直接加载完整训练权重：

```python
model_path = "logs/xxx/best_epoch_weights.pth"
```

必须保证：

```text
num_classes
phi
in_channels
input_shape
```

和训练时一致。

## 5. 训练

训练入口：

```bash
python train.py
```

当前默认关键参数：

```python
num_classes = 2
phi = "b0"
input_shape = [256, 256]
```

训练时会在配置表里打印：

```text
band_mode
image_ext
selected_bands
in_channels
normalization_config
```

每次训练前建议确认这些值是否和当前实验一致。

## 6. 单张预测

预测入口：

```bash
python predict.py
```

当 `image_ext = ".tif"` 时，输入路径可以是 tif：

```text
VOCdevkit/VOC2007/JPEGImages/000001.tif
```

`predict.py` 会通过：

```python
segformer.open_image(path)
```

自动判断 tif / jpg。多光谱 tif 会用 `rasterio` 读取 `selected_bands`。

可视化叠加时，模型仍然使用全部输入波段；显示底图使用：

```python
vis_bands = [3, 2, 1]
```

即真彩色 `[B4, B3, B2]`。

## 7. mIoU 评估

多光谱评估脚本：

```bash
python 多光谱get_miou.py
```

它会：

```text
1. 读取 val.txt
2. 读取 JPEGImages/xxx.tif
3. 生成预测 mask 到 miou_out_multispectral/detection-results/
4. 与 SegmentationClass/xxx.png 计算 mIoU
```

脚本中的模式：

```python
miou_mode = 0  # 生成预测 + 计算 mIoU
miou_mode = 1  # 只生成预测
miou_mode = 2  # 只计算 mIoU
```

## 8. 模型结构统计

统计入口：

```bash
python summary.py
```

它会读取当前 `multispectral_config.py` 中的 `in_channels`。

如果当前环境缺依赖，先安装：

```bash
pip install thop torchsummary
```

## 9. 依赖

多光谱版本额外需要：

```text
rasterio
```

模型统计需要：

```text
thop
torchsummary
```

这些已经写入 `requirements.txt`。

## 10. 推荐实验顺序

建议先做基础对比：

```text
SegFormer-B0 RGB
SegFormer-B0 4band
SegFormer-B0 6band
```

如果显存和时间允许，再做：

```text
SegFormer-B1 6band
SegFormer-B2 6band
```

基础版不建议一开始用 B4/B5，模型更重，训练成本高，数据少时也更容易过拟合。
