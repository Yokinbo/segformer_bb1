#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from multispectral_config import band_mode, in_channels, selected_bands
from nets.segformer import SegFormer

if __name__ == "__main__":
    # 这里与 train.py 保持一致，用 multispectral_config.py 控制输入通道数。
    # 例如：
    # - band_mode="rgb"   -> in_channels=3
    # - band_mode="4band" -> in_channels=4
    # - band_mode="6band" -> in_channels=6
    input_shape     = [256, 256]
    num_classes     = 2
    phi             = 'b0'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = SegFormer(num_classes = num_classes, phi = phi, pretrained=False, in_channels=in_channels).to(device)
    print("band_mode:", band_mode)
    print("selected_bands:", selected_bands)
    print("in_channels:", in_channels)
    summary(model, (in_channels, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
