# 文件: models/dnf_model.py (Jittor版本)

from jittor import nn # Jittor改动：这是唯一需要修改的行

# Jittor改动：下面的导入现在会引入我们已经翻译好的Jittor模块
from .dnf_utils.base import DNFBase
from .dnf_utils.cid import CID
from .dnf_utils.mcc import MCC
from .dnf_utils.fuse import PDConvFuse, GFM
from .dnf_utils.resudual_switch import ResidualSwitchBlock
from utils.registry import MODEL_REGISTRY
from .utils import conv_with_padding_mode

# @MODEL_REGISTRY.register()
# class SingleStageNet(DNFBase):
#     def __init__(self, f_number, *, 
#                  block_size=1, 
#                  layers=4, 
#                  denoising_block='CID', 
#                  color_correction_block='MCC') -> None:
#         super().__init__(f_number, block_size=block_size, layers=layers, 
#                          denoising_block=denoising_block, color_correction_block=color_correction_block)
    
#     def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
#         return x, None, encoder_features


# @MODEL_REGISTRY.register()
# class MultiStageNet(DNFBase):
#     def __init__(self, f_number, *, 
#                  block_size=1, 
#                  layers=4, 
#                  denoising_block='CID', 
#                  color_correction_block='MCC') -> None:
#         super().__init__(f_number, block_size=block_size, layers=layers, 
#                          denoising_block=denoising_block, color_correction_block=color_correction_block)

#         aux_outchannel = 3 if block_size == 1 else block_size * block_size
#         self.aux_denoising_blocks = nn.ModuleList([
#             ResidualSwitchBlock(
#                 self.denoising_block_class(
#                     f_number=f_number * (2**idx),
#                     padding_mode=self.padding_mode
#                 )
#             )
#             for idx in range(layers)
#         ])
#         self.aux_upsamples = nn.ModuleList([
#             self.upsample_class(
#                 f_number * (2**idx), 
#                 padding_mode=self.padding_mode
#             )
#             for idx in range(1, layers)
#         ])
#         self.denoising_decoder_fuses = nn.ModuleList([
#             self.decoder_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers - 1)
#         ])

#         self.aux_conv_fuse_0 = conv_with_padding_mode(f_number, f_number, 3, 1, 1, bias=True, padding_mode=self.padding_mode)
#         self.aux_conv_fuse_1 = nn.Conv2d(f_number, aux_outchannel, 1, 1, 0, bias=True)
        
#         inchannel = 3 if block_size == 1 else block_size * block_size
#         self.aux_feature_conv_0 = conv_with_padding_mode(inchannel, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
#         self.aux_feature_conv_1 = conv_with_padding_mode(f_number, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
        
#         head = [2 ** layer for layer in range(layers)]
#         self.aux_color_correction_blocks = nn.ModuleList([
#             self.color_correction_block_class(
#                 f_number=f_number * (2 ** idx),
#                 num_heads=head[idx],
#                 padding_mode=self.padding_mode,
#             )
#             for idx in range(layers)
#         ])
#         self.aux_downsamples = nn.ModuleList([
#             self.downsample_class(
#                 f_number * (2**idx), 
#                 padding_mode=self.padding_mode
#             )
#             for idx in range(layers - 1)
#         ])
        
#     def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
#         denoise_decoder_features = []
#         for denoise, up, fuse, encoder_feature in reversed(list(zip(
#             self.aux_denoising_blocks[1:], 
#             self.aux_upsamples, 
#             self.denoising_decoder_fuses,
#             encoder_features    
#         ))):
#             x = denoise(x, 1)
#             denoise_decoder_features.append(x)
#             x = up(x)
#             x = fuse(x, encoder_feature)
#         x = self.aux_denoising_blocks[0](x, 1)
#         denoise_decoder_features.append(x)
#         x = x + f_short_cut
#         x = self.act(self.aux_conv_fuse_0(x))
#         x = self.aux_conv_fuse_1(x)
#         res1 = x

#         encoder_features = []
#         x = self.act(self.aux_feature_conv_0(res1))
#         x = self.aux_feature_conv_1(x)
#         for color_correction, down in zip(self.aux_color_correction_blocks[:-1], self.aux_downsamples):
#             x = color_correction(x)
#             encoder_features.append(x)
#             x = down(x)
#         x = self.aux_color_correction_blocks[-1](x)
#         return x, res1, encoder_features


@MODEL_REGISTRY.register()
class DNF(DNFBase):
    def __init__(self, f_number, *,
                 block_size=1,
                 layers=4,
                 denoising_block='CID',
                 color_correction_block='MCC',
                 feedback_fuse='GFM'
                 ) -> None:
        super(DNF, self).__init__(f_number=f_number, block_size=block_size, layers=layers,
                                    denoising_block=denoising_block, color_correction_block=color_correction_block)
        def get_class(class_or_class_str):
            return eval(class_or_class_str) if isinstance(class_or_class_str, str) else class_or_class_str
        
        self.feedback_fuse_class = get_class(feedback_fuse)

        self.feedback_fuses = nn.ModuleList([
            self.feedback_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers)
        ])

        # 这是辅助解码器三件套：一个层级一个 CID（包在残差开关块里，这里用 switch=1，因为要重建），
        # 上采样把分辨率放大，PDConvFuse 与第一遍编码缓存的跳连 encoder_features 融合。
        aux_outchannel = 3 if block_size == 1 else block_size * block_size
        self.aux_denoising_blocks = nn.ModuleList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2**idx),
                    padding_mode=self.padding_mode
                )   
            )
            for idx in range(layers)
        ])
        self.aux_upsamples = nn.ModuleList([
            self.upsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(1, layers)
        ])
        self.denoising_decoder_fuses = nn.ModuleList([
            self.decoder_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers - 1)
        ])

        self.aux_conv_fuse_0 = conv_with_padding_mode(f_number, f_number, 3, 1, 1, bias=True, padding_mode=self.padding_mode)
        self.aux_conv_fuse_1 = nn.Conv2d(f_number, aux_outchannel, 1, 1, 0, bias=True)

    def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
        ## denoising decoder  # 辅助去噪解码器 (生成去噪先验)
        denoise_decoder_features = []
        for denoise, up, fuse, encoder_feature in reversed(list(zip(
            self.aux_denoising_blocks[1:], 
            self.aux_upsamples, 
            self.denoising_decoder_fuses,
            encoder_features    
        ))):
            x = denoise(x, 1)
            # 每一层解码器重建的特征 x 被存入 denoise_decoder_features 列表。这些就是将要被反馈的特征。
            denoise_decoder_features.append(x)  
            x = up(x)
            x = fuse(x, encoder_feature)
        x = self.aux_denoising_blocks[0](x, 1)
        denoise_decoder_features.append(x)
        x = x + f_short_cut
        x = self.act(self.aux_conv_fuse_0(x))
        x = self.aux_conv_fuse_1(x)
        res1 = x

 
        ## feedback, local residual switch on # 反馈与二次编码 (应用去噪先验)
        # 利用刚刚生成的 denoise_decoder_features 作为先验知识，对原始输入进行一次更精准的、以信号重建为目的的编码过程。
        encoder_features = []
        denoise_decoder_features.reverse()
        x = f_short_cut
        for i, (fuse, denoise, down, decoder_feedback_feature) in enumerate(zip(
            self.feedback_fuses[:-1], 
            self.denoising_blocks[:-1], 
            self.downsamples,
            denoise_decoder_features[:-1]
        )):
            
        # ----> 在这里加入详细的打印 <----
            # 为了避免刷屏，我们可以只在训练的第一个batch打印一次
            # if not hasattr(self, '_has_printed_stats'):
            #     print(f"\n--- Debugging feedback_fuses[{i}] ---")
            #     print(f"Input 'x' stats:          shape={x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            #     print(f"Input 'feedback' stats:   shape={decoder_feedback_feature.shape}, mean={decoder_feedback_feature.mean().item():.4f}, std={decoder_feedback_feature.std().item():.4f}")
                
            #     # 打印 fuse 模块某个权重的标准差，用来观察它是否变化
            #     weight_std = fuse.parameters()[0].std().item()
            #     print(f"Fuse module weight std:   {weight_std:.4f}")


            #  GFM (Gated Fusion Module) 的实际调用 。它将当前层的输入特征 x 与来自辅助解码器的反馈特征 decoder_feedback_feature 进行智能融合
            x = fuse(x, decoder_feedback_feature)    


            
            # if not hasattr(self, '_has_printed_stats'):
            #     print(f"Output 'x' stats:         shape={x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            #     print(f"--- End Debugging ---\n")

            x = denoise(x, 1)
            encoder_features.append(x)
            x = down(x)

        # # 在循环结束后，设置一个标记，确保只打印一次
        # if not hasattr(self, '_has_printed_stats'):
        #     self._has_printed_stats = True


        x = self.feedback_fuses[-1](x, denoise_decoder_features[-1])
        x = self.denoising_blocks[-1](x, 1)  # residual switch on
        
        return x, res1, encoder_features