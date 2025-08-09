# 文件: forwards/single_stage_forward.py (Jittor版本)

from utils.registry import FORWARD_REGISTRY
# Jittor改动: 移除了 from fvcore.nn import FlopCountAnalysis, flop_count_table

@FORWARD_REGISTRY.register()
def ss_train_forward(config, model, data):
    # Jittor改动: 无需手动 .cuda()，Jittor会自动处理
    raw = data['noisy_raw']
    rgb_gt = data['clean_rgb']
    
    rgb_out = model(raw)
    ###### | output         | label
    return {'rgb': rgb_out}, {'rgb': rgb_gt}


@FORWARD_REGISTRY.register()
def ss_test_forward(config, model, data):
    # Jittor改动: 无需 .cuda()，也不再需要if/else来区分cpu/gpu
    raw = data['noisy_raw']
    rgb_gt = data['clean_rgb']
    img_files = data['img_file']
    lbl_files = data['lbl_file']

    rgb_out = model(raw)

    return {'rgb': rgb_out}, {'rgb': rgb_gt}, img_files, lbl_files


@FORWARD_REGISTRY.register()  # without label, for inference only
def ss_inference(config, model, data):
    # Jittor改动: 无需手动 .cuda()
    raw = data['noisy_raw']
    img_files = data['img_file']

    rgb_out = model(raw)

    ###### | output         | img names
    return {'rgb': rgb_out}, img_files