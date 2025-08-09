# 文件: utils/scheduler.py (Jittor版本)

# Jittor改动：从 jittor.lr_scheduler 导入对应的调度器
from jittor.lr_scheduler import CosineAnnealingLR, StepLR

def build_scheduler(config, optimizer, n_iter_per_epoch=1):
    # Jittor改动：这部分计算总步数的逻辑保持不变
    if config['lr_scheduler']['t_in_epochs']:
        n_iter_per_epoch = 1
    num_steps = int(config['epochs'] * n_iter_per_epoch)
    warmup_steps = int(config['warmup_epochs'] * n_iter_per_epoch)
    
    lr_scheduler = None
    scheduler_type = config['lr_scheduler']['type']

    if scheduler_type == 'cosine':
        # Jittor改动：使用 jittor.lr_scheduler.CosineAnnealingLR
        # 注意：Jittor内置的调度器不直接包含warmup，warmup需在训练循环中单独实现
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_steps - warmup_steps, # 对应timm的t_initial，且需减去warmup步数
            eta_min=config['min_lr'],      # 对应timm的lr_min
        )
    elif scheduler_type == 'step':
        # Jittor改动：使用 jittor.lr_scheduler.StepLR
        decay_steps = int(config['lr_scheduler']['decay_epochs'] * n_iter_per_epoch)
        lr_scheduler = StepLR(
            optimizer,
            step_size=decay_steps, # 对应timm的decay_t
            gamma=config['lr_scheduler']['decay_rate'], # 对应timm的decay_rate
        )
    else:
        raise NotImplementedError(f"Scheduler '{scheduler_type}' not implemented for Jittor.")
    
    # 注意：对于需要warmup的情况，您需要在runner.py的训练循环中添加手动控制逻辑
    # 例如:
    # if epoch < warmup_epochs:
    #     # 手动设置warmup期间的学习率
    # else:
    #     lr_scheduler.step()

    return lr_scheduler