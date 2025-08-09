# 文件: forwards/dnf_forward.py (Jittor版本)

from utils.registry import FORWARD_REGISTRY
# Jittor改动: 导入jittor用于性能分析
import jittor as jt

@FORWARD_REGISTRY.register(suffix='DNF')
def train_forward(config, model, data):
    # Jittor改动: 无需手动.cuda()，Jittor会自动处理设备
    raw = data['noisy_raw']
    raw_gt = data['clean_raw']
    rgb_gt = data['clean_rgb']
    
    rgb_out, raw_out = model(raw)
    
    ###### | output                         | label
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}


@FORWARD_REGISTRY.register(suffix='DNF')
def test_forward(config, model, data):
    # Jittor改动: 无需手动.cuda()，也不再需要if/else来区分cpu/gpu
    raw = data['noisy_raw']
    raw_gt = data['clean_raw']
    rgb_gt = data['clean_rgb']
    img_files = data['img_file']
    lbl_files = data['lbl_file']

    rgb_out, raw_out = model(raw)

    ###### | output                         | label                              | img and label names
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}, img_files, lbl_files


@FORWARD_REGISTRY.register(suffix='DNF')  # without label, for inference only
def inference(config, model, data):
    # Jittor改动: 无需手动.cuda()
    raw = data['noisy_raw']
    img_files = data['img_file']

    rgb_out, raw_out = model(raw)

    ###### | output                         | img names
    return {'rgb': rgb_out, 'raw': raw_out}, img_files



# 注意这个函数！！！这个函数错了很多次
@FORWARD_REGISTRY.register()
def DNF_profile(config, model, data, logger):
    """
    使用Jittor的性能分析工具jt.profile_scope来分析模型的计算量和内存占用。
    """
    # 1. 准备输入数据
    print("准备开始profile")
    x = data['noisy_raw']

    # 2. 使用 jt.profile_scope 进行性能分析
    # 原来的 jt.profile 函数可能因内部实现或环境问题导致异常。
    # 我们改用 jt.profile_scope 上下文管理器，这是更稳定和推荐的方式。

        # 在 profile_scope 上下文中执行模型的前向传播
    with jt.profile_scope() as report:
        print(report)