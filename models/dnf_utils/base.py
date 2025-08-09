# 文件：models/base.py (Jittor版本)

from jittor import nn # Jittor改动
import jittor as jt # Jittor改动：为了使用 jt.nn.pad

# Jittor改动：确保你已经将这些文件翻译成了Jittor版本
from .fuse import PDConvFuse
from .cid import CID
from .mcc import MCC
from .sample import SimpleDownsample, SimpleUpsample
from .resudual_switch import ResidualSwitchBlock

from abc import abstractmethod

# Jittor改动：添加这个辅助函数
from ..utils import conv_with_padding_mode


class DNFBase(nn.Module):
    def __init__(self, f_number, *,
                 block_size=1,
                 layers=4,
                 denoising_block='CID',
                 color_correction_block='MCC'
                 ) -> None:
        super().__init__()
        def get_class(class_or_class_str):
            # 这里的eval是原代码设计，保持不变
            return eval(class_or_class_str) if isinstance(class_or_class_str, str) else class_or_class_str

        self.denoising_block_class = get_class(denoising_block)
        self.color_correction_block_class = get_class(color_correction_block)
        self.downsample_class = SimpleDownsample
        self.upsample_class = SimpleUpsample
        self.decoder_fuse_class = PDConvFuse

        self.padding_mode = 'reflect'
        self.act = nn.GELU()
        self.layers = layers

        head = [2 ** layer for layer in range(layers)]
        self.block_size = block_size
        inchannel = 3 if block_size == 1 else block_size * block_size
        outchannel = 3 * block_size * block_size

        # 在 PyTorch 中，nn.Conv2d 层自带一个非常方便的 padding_mode 参数，可以直接指定 'zeros', 'reflect' (反射填充), 'replicate' (复制填充) 等不同的填充方式。

        # 在 Jittor 中，nn.Conv2d 没有这个 padding_mode 参数。它只支持默认的零填充（通过 padding 参数指定大小）。对于其他类型的填充，Jittor要求您使用一个独立的、显式的填充层来完成

        # self.feature_conv_0 = nn.Conv2d(inchannel, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
        # self.feature_conv_1 = nn.Conv2d(f_number, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
        
        # 两个卷积层都使用了5x5的卷积核，padding=2，这样可以保持输入输出的空间尺寸不变
        self.feature_conv_0 = conv_with_padding_mode(inchannel, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
        self.feature_conv_1 = conv_with_padding_mode(f_number, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)

        self.downsamples = nn.ModuleList([
            self.downsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(layers - 1)
        ])

        self.upsamples = nn.ModuleList([
            self.upsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(1, layers)
        ])

        # 编码器组件 # 噪声估计阶段  # CID + ResidualSwitchBlock
        # ResidualSwitchBlock 是一个残差开关块，它可以根据输入的开
        self.denoising_blocks = nn.ModuleList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2**idx),
                    padding_mode=self.padding_mode
                )
            )
            for idx in range(layers)
        ])
            
         
        # 解码器组件 # 色彩恢复阶段 # MCC（颜色校正）
        self.color_correction_blocks = nn.ModuleList([
            self.color_correction_block_class(
                f_number=f_number * (2 ** idx),
                num_heads=head[idx],
                padding_mode=self.padding_mode,
            )
            for idx in range(layers)
        ])

        self.color_decoder_fuses = nn.ModuleList([
            self.decoder_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers - 1)
        ])

        self.conv_fuse_0 = conv_with_padding_mode(f_number, f_number, 3, 1, 1, bias=True, padding_mode=self.padding_mode)
        self.conv_fuse_1 = nn.Conv2d(f_number, outchannel, 1, 1, 0, bias=True)

        if block_size > 1:
            self.pixel_shuffle = nn.PixelShuffle(block_size)
        else:
            self.pixel_shuffle = nn.Identity()

    @abstractmethod  # 和c++里的 virtual 一样
    def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
        pass

    #  确保输入尺寸可以被完整地降采样和上采样
    def _check_and_padding(self, x):
        _, _, h, w = x.size()
        stride = (2 ** (self.layers - 1)) # 

        dh = -h % stride
        dw = -w % stride

        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w + left_pad, top_pad, h + top_pad)

        # Jittor改动: F.pad 替换为 jt.nn.pad
        padded_tensor = jt.nn.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )
        return padded_tensor
        
    # 将输出裁剪回原始尺寸
    def _check_and_crop(self, x, res1):
        # 此函数使用张量切片，Jittor与PyTorch语法相同，无需改动
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top*self.block_size:bottom*self.block_size, left*self.block_size:right*self.block_size]
        res1 = res1[:, :, top:bottom, left:right] if res1 is not None else None
        return x, res1

# Jittor 框架规定，所有继承自 jt.Module 的模型，其前向传播方法必须命名为 execute，而不是 PyTorch 中使用的 forward
    def execute(self, x):
        # forward函数中的逻辑流程与PyTorch完全相同，无需改动
        x = self._check_and_padding(x)
        x = self.act(self.feature_conv_0(x))
        x = self.feature_conv_1(x)
        f_short_cut = x

        ## encoder, local residual switch off  噪声估计阶段
        encoder_features = []

        # zip函数：将多个列表中对应位置的元素打包成一个个元组 (tuple)
        for denoise, down in zip(self.denoising_blocks[:-1], self.downsamples):
            x = denoise(x, 0)  # 残差开关设为关闭
            encoder_features.append(x)
            x = down(x)
        x = self.denoising_blocks[-1](x, 0)
        # 处理的是U-Net结构最底部的瓶颈层（Bottleneck），即最后一个去噪层。它没有对应的下采样层，所以被单独处理。

        x, res1, refined_encoder_features = self._pass_features_to_color_decoder(x, f_short_cut, encoder_features) 

        ## Decoder，色彩恢复阶段
        for color_correction, up, fuse, encoder_feature in reversed(list(zip(
            self.color_correction_blocks[1:], 
            self.upsamples, 
            self.color_decoder_fuses,
            refined_encoder_features  # 接收的跳跃连接特征是经过反馈提炼后的 refined_encoder_features，而不是初始的 encoder_features
        ))):
            x = color_correction(x) # MCC块
            x = up(x) # 上采样
            x = fuse(x, encoder_feature)
        x = self.color_correction_blocks[0](x)

        x = self.act(self.conv_fuse_0(x))
        x = self.conv_fuse_1(x)
        x = self.pixel_shuffle(x)
        rgb, raw = self._check_and_crop(x, res1)
        return rgb, raw