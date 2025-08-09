# 文件: models/dnf_utils/resudual_switch.py (Jittor版本)
from jittor import nn # Jittor改动

class ResidualSwitchBlock(nn.Module):
    def __init__(self, block) -> None:
        super().__init__()
        self.block = block
        
    def execute(self, x, residual_switch):
        return self.block(x) + residual_switch * x