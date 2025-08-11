# 文件: models/dnf_utils/fuse.py (Jittor版本)
import jittor as jt
from jittor import nn
from ..utils import conv_with_padding_mode

class PDConvFuse(nn.Module):
    def __init__(self, in_channels=None, f_number=None, feature_num=2, bias=True, **kwargs) -> None:
        super().__init__()
        if in_channels is None:
            assert f_number is not None
            in_channels = f_number
        self.feature_num = feature_num
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(feature_num * in_channels, in_channels, 1, 1, 0, bias=bias)
        self.dwconv = conv_with_padding_mode(in_channels, in_channels, 3, 1, 1, bias=bias, groups=in_channels, padding_mode='reflect')

    def execute(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        # Jittor改动: torch.cat -> jt.concat
        concated_feats = jt.concat(inp_feats, dim=1)
        return self.dwconv(self.act(self.pwconv(concated_feats)))

class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs) -> None:
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
        self.dwconv = conv_with_padding_mode(hidden_features * 2, hidden_features * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

    def execute(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        shortcut = inp_feats[0]
        # Jittor改动: torch.cat -> jt.concat
        x = jt.concat(inp_feats, dim=1)
        x = self.pwconv(x)
        x = self.dwconv(x)                       # ✅ 关键补上这一步
        # Jittor改动: .chunk在Jittor中同样可用
        x1, x2 = jt.chunk(x, 2, dim=1)
        # Jittor改动: F.gelu -> jt.nn.gelu
        x = jt.nn.gelu(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)