# 文件: models/utils.py (请确保这是文件的全部内容)

import jittor as jt
from jittor import nn

# --- 第一部分：Jittor 版本的 LayerNorm ---
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(jt.ones(normalized_shape))
        self.bias = nn.Parameter(jt.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def execute(self, x):
        if self.data_format == "channels_last":
            return jt.nn.layer_norm(x, self.normalized_shape, self.eps, self.weight, self.bias)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / jt.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# --- 第二部分：必须包含的新增辅助函数 ---
def conv_with_padding_mode(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros', groups=1):
    """
    一个辅助函数，用于创建带特定padding_mode的卷积层。
    它将一个Padding层和一个padding=0的Conv层打包成一个nn.Sequential。
    """
    if padding_mode == 'reflect':
        pad_layer = nn.ReflectionPad2d(padding)
        conv_padding = 0
    elif padding_mode == 'replicate':
        pad_layer = nn.ReplicationPad2d(padding)
        conv_padding = 0
    else:  # 默认为 'zeros'
        pad_layer = nn.Identity()
        conv_padding = padding

    return nn.Sequential(
        pad_layer,
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=conv_padding, bias=bias, groups=groups)
    )