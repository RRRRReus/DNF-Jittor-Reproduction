# 文件: utils/optimizer.py (Jittor版本)

from jittor import optim # Jittor改动：将torch.optim替换为jittor.optim

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    
    # Jittor改动：set_weight_decay函数本身无需改动，因为它处理的是通用逻辑
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config['optimizer']['type'].lower()
    optimizer = None
    if opt_lower == 'sgd':
        # Jittor改动：optim.SGD现在是jittor.optim.SGD
        optimizer = optim.SGD(parameters, momentum=config['optimizer']['momentum'], nesterov=True,
                              lr=config['base_lr'], weight_decay=config['weight_decay'])
    elif opt_lower == 'adamw':
        # Jittor改动：optim.AdamW现在是jittor.optim.AdamW
        optimizer = optim.AdamW(parameters, eps=config['optimizer']['eps'], betas=config['optimizer']['betas'],
                                lr=config['base_lr'], weight_decay=config['weight_decay'])

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    # --- 这个函数无需任何改动 ---
    # Jittor的模型参数遍历(model.named_parameters())和参数属性(.requires_grad, .shape)
    # 与PyTorch完全相同。将参数分组传递给优化器的方式也完全相同。
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
            
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    # --- 这个函数是纯Python，无需任何改动 ---
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin