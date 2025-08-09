# 文件: models/dnf_utils/sample.py (Jittor版本)
from jittor import nn # Jittor改动
from ..utils import conv_with_padding_mode

class SimpleDownsample(nn.Module):
    def __init__(self, dim, *, padding_mode='reflect'):
        super().__init__()
        self.body = conv_with_padding_mode(
                    dim, dim*2, kernel_size=2, stride=2, padding=0, bias=False, 
                    padding_mode=padding_mode)
        # 一个 k=2, s=2 的卷积（没有零填充），直接把特征图的 H、W 各减半，同时把 通道数从 dim 提升到 2×dim

    def execute(self, x):
        return self.body(x)

class SimpleUpsample(nn.Module):
    def __init__(self, dim, *, padding_mode='reflect'):
        super().__init__()
        self.body = nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2, padding=0, bias=False)
        # 一个 反卷积（ConvTranspose2d），同样用 k=2, s=2，把特征图的 H、W 各×2，同时把 通道数从 dim 降到 dim/2


    def execute(self, x):
        return self.body(x)