# 文件: utils/miscs.py (Jittor版本)

import os
import shutil
import cv2
import numpy as np
import jittor as jt # Jittor改动

def load_checkpoint(config, model, optimizer, lr_scheduler, logger, epoch=None):
    resume_ckpt_path = config['train']['resume']
    logger.info(f"==============> Resuming form {resume_ckpt_path}....................")
    
    # Jittor改动: Jittor没有内置的torch.hub，从URL加载需要自定义实现  # 潜在的问题
    # 这里我们假设检查点是本地文件
    if resume_ckpt_path.startswith('https'):
        raise NotImplementedError("Loading checkpoint from URL is not supported by default in Jittor. Please download the file first.")
    else:
        # Jittor改动: torch.load -> jt.load
        checkpoint = jt.load(resume_ckpt_path)
        
    # Jittor改动: model.load_state_dict -> model.load
    # Jittor的model.load默认就是strict=False的行为
    all_keys = set(checkpoint['model'].keys())
    # model.load(checkpoint['model'])
    model.load_parameters(checkpoint['model']) # Jittor改动：使用 load_parameters 从字典加载
    logger.info(f"<All keys matched successfully>")

    max_psnr = 0.0
    if not config.get('eval_mode', False) and 'optimizer' in checkpoint and 'lr_scheduler_last_epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.last_epoch = checkpoint['lr_scheduler_last_epoch'] # 由 load_state_dict <--- 正确的行
        if 'max_psnr' in checkpoint:
            max_psnr = checkpoint['max_psnr']
            
    if epoch is None and 'epoch' in checkpoint:
        config['train']['start_epoch'] = checkpoint['epoch']
        logger.info(f"=> loaded successfully '{resume_ckpt_path}' (epoch {checkpoint['epoch']})")
        
    del checkpoint
    # Jittor改动: torch.cuda.empty_cache() -> jt.clean_graph()
    jt.clean_graph()
    return max_psnr

def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config['train']['pretrained']}....................")
    # Jittor改动: torch.load -> jt.load
    checkpoint = jt.load(config['train']['pretrained'])
    
    # Jittor改動：Jittor中模型权重通常直接保存在顶层
    state_dict = checkpoint.get('model', checkpoint)
    
    # Jittor改动: model.load_state_dict -> model.load
    # model.load(state_dict)
    
    model.load_parameters(state_dict) # Jittor改动：使用 load_parameters 从字典加载
    logger.info(f"<All keys matched successfully>")
    logger.info(f"=> loaded successfully '{config['train']['pretrained']}'")

    del checkpoint
    jt.clean_graph()
    
def save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, logger, is_best=False):
    # Jittor改动: .state_dict() API 相同
    save_state = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler_last_epoch': lr_scheduler.last_epoch, # 由 lr_scheduler.state_dict() 得到
                'max_psnr': max_psnr,
                'epoch': epoch,
                'config': config}

    os.makedirs(os.path.join(config['output'], 'checkpoints'), exist_ok=True)

    # Jittor改动：Jittor约定俗成使用.pkl或.bin作为后缀
    save_path = os.path.join(config['output'], 'checkpoints', 'checkpoint.pkl')
    logger.info(f"{save_path} saving......")
    # Jittor改动: torch.save -> jt.save
    jt.save(save_state, save_path)
    logger.info(f"{save_path} saved")

    # Jittor改动：修改后缀名
    if epoch % config['save_per_epoch'] == 0 or (config['train']['epochs'] - epoch) < 50:
        shutil.copy(save_path, os.path.join(config['output'], 'checkpoints', f'epoch_{epoch:04d}.pkl'))
        logger.info(f"{save_path} copied to epoch_{epoch:04d}.pkl")
    if is_best:
        shutil.copy(save_path, os.path.join(config['output'], 'checkpoints', 'model_best.pkl'))
        logger.info(f"{save_path} copied to model_best.pkl")
        
def get_grad_norm(parameters, optimizer, norm_type=2):
    # Jittor改动：需要重写梯度范数的计算逻辑
    if isinstance(parameters, jt.Var):
        parameters = [parameters]


    # parameters = list(filter(lambda p: p.grad is not None, parameters))
    parameters = list(filter(lambda p: p.opt_grad(optimizer) is not None, parameters))

    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        # 在 PyTorch 中,当您调用loss.backward()后，每个参数的梯度会被计算并存储在.grad属性中，您可以通过p.grad来直接访问
        # 在 Jittor 中，梯度是在optimizer.step(loss)这一步中计算的,您不能再用p.grad来获取梯度，而必须使用Jittor提供的新接口：p.opt_grad(optimizer)
        # param_norm_val = (p.grad.abs()**norm_type).sum()
        param_norm_val = (p.opt_grad(optimizer).abs()**norm_type).sum()
        total_norm += param_norm_val.item()
    
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def save_image_jittor(img, file_path, range_255_float=True, params=None, auto_mkdir=True):
    # Jittor改动：函数重命名，并修改内部实现
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    
    # Jittor改动: .size() -> .shape
    assert len(img.shape) == 3
    # Jittor改动: .cpu().detach()不再需要，直接.numpy()
    img = img.clone().numpy().transpose(1, 2, 0)

    # 以下Numpy和OpenCV操作与框架无关，保持不变
    if range_255_float:
        img = img.clip(0, 255).round()
        img = img.astype(np.uint8)
    else:
        img = img.clip(0, 1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')