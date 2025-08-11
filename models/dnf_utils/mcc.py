# 文件: models/dnf_utils/mcc.py (Jittor版本)
from einops import rearrange
import jittor as jt
from jittor import nn

from ..utils import LayerNorm # 确保这个文件已经被翻译
from ..utils import conv_with_padding_mode

class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')

        self.num_heads = num_heads
        # Jittor改动: 使用jt.ones创建Parameter
        self.temperature = jt.ones((num_heads, 1, 1))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = conv_with_padding_mode(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.GELU(),
            conv_with_padding_mode(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.GELU()
        )

    # 替换后的 execute 函数
    def execute(self, x):
        attn = self.norm(x)
        b, c, h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
        # Jittor改动: 使用jt.chunk
        q, k, v = jt.chunk(qkv, 3, dim=1)

        # ---- Jittor原生实现替换 einops.rearrange ----
        # 计算每个头的通道数 c_per_head
        c_per_head = c // self.num_heads
        
        # 原: q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = q.reshape(b, self.num_heads, c_per_head, h * w)

        # 原: k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = k.reshape(b, self.num_heads, c_per_head, h * w)

        # 原: v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = v.reshape(b, self.num_heads, c_per_head, h * w)
        # -------------------------------------------

        # Jittor改动: torch.nn.functional.normalize -> jt.normalize
        q = jt.normalize(q, dim=-1, p=2)
        k = jt.normalize(k, dim=-1, p=2)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        # ---- Jittor原生实现替换 einops.rearrange (逆过程) ----
        # 原: out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.reshape(b, c, h, w)
        # ----------------------------------------------------

        out = self.project_out(out)
        out = self.feedforward(out + x)
        return out