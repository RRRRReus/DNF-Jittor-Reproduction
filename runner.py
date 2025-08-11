# 文件：runner.py (Jittor完整版)

import os
import time
import datetime
import yaml
import git

import jittor as jt
# Jittor修正：从本地utils导入AverageMeter，移除timm依赖
from utils.meters import AverageMeter 
# Jittor修正：使用tensorboardX替代torch.utils.tensorboard
from tensorboardX import SummaryWriter

from utils import (load_checkpoint, load_pretrained, save_checkpoint, 
                   save_image_jittor, get_grad_norm, get_psnr_jittor, get_ssim_jittor)
from utils.config import parse_options, copy_cfg, ordered_dict_to_dict
from utils.scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.loss import build_loss
from utils.logger import create_logger

from models import build_model
from datasets import build_train_loader, build_valid_loader, build_test_loader
from forwards import build_forwards, build_profile

def main(config):

    # config是一个字典
    writer = SummaryWriter(os.path.join(config['output'], 'tensorboard'))
    train_dataloader = build_train_loader(config['data'])
    if not config['testset_as_validset']:
        valid_dataloader = build_valid_loader(config['data'], 1)
    else:
        valid_dataloader = build_test_loader(config['data'], 2)
    
    logger.info("data加载完成")

    logger.info(f"Creating model:{config['name']}/{config['model']['type']}")
    model = build_model(config['model'])
    # logger.info(str(model)) 
    profile_forward = build_profile(config)
    profile_model(config, profile_forward, model, train_dataloader, logger)

    optimizer = build_optimizer(config['train'], model)
    lr_scheduler = build_scheduler(config['train'], optimizer, len(train_dataloader))
    loss_list = build_loss(config['loss'])
    logger.info(str(loss_list))
    
    logger.info('Building forwards:')
    logger.info(f'Train forward: {config["train"]["forward_type"]}')
    logger.info(f'Test forward: {config["test"]["forward_type"]}')
    train_forward, test_forward = build_forwards(config)


    # 检查训练模式
    max_psnr = 0.0
    max_ssim = 0.0
    total_epochs = config['train']['early_stop'] if config['train']['early_stop'] is not None else config['train']['epochs']

    if config.get('throughput_mode', False):
        throughput(config, train_forward, model, valid_dataloader, logger)
        return

    if config['train']['auto_resume']:
        auto_resume_path = os.path.join(config['output'], 'checkpoints', 'checkpoint.pkl') # Jittor改动：后缀名建议
        if os.path.exists(auto_resume_path):
            config['train']['resume'] = auto_resume_path
            logger.info(f'Auto resume: setting resume path to {auto_resume_path}')
    
    if config['train'].get('resume'):
        max_psnr = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        validate(config, test_forward, model, loss_list, valid_dataloader, config['train'].get('start_epoch', 0), writer)
        if config.get('eval_mode', False):
            return

    if config['train'].get('pretrained') and (not config['train'].get('resume')):
        load_pretrained(config, model, logger)
        validate(config, test_forward, model, loss_list, valid_dataloader, config['train'].get('start_epoch', 0), writer)
        if config.get('eval_mode', False):
            return


    # 开始训练
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config['train'].get('start_epoch', 0)+1, total_epochs+1):
        # # Jittor改动：学习率调度器的step调用放在循环内，与PyTorch习惯保持一致

        # 在 PyTorch/timm 中，调度器的step方法通常需要您传入当前的epoch号
        # 无需传入任何参数，只需在每个epoch结束时调用一次lr_scheduler.step()
        # if not config['train']['lr_scheduler']['t_in_epochs']:
        #     pass # 迭代步进的调度器在train_one_epoch中处理
        # else:
        #     lr_scheduler.step(epoch-1) # 假设您的scheduler在epoch开始前更新

        # 注意这个地方需要补充warmup吗

        train_one_epoch(config, train_forward, model, loss_list, train_dataloader, optimizer, None, epoch, lr_scheduler, writer)
        # Jittor改动：在每个epoch的训练结束后调用step()，且不带任何参数
        if config['train']['lr_scheduler']['t_in_epochs']:
            lr_scheduler.step()


        if epoch % config['valid_per_epoch'] == 0 or (total_epochs - epoch) < 50:
            psnr, ssim, loss = validate(config, test_forward, model, loss_list, valid_dataloader, epoch, writer)
            is_best = psnr > max_psnr
            max_psnr = max(max_psnr, psnr)
            max_ssim = max(max_ssim, ssim) if is_best else max_ssim
            writer.add_scalar('eval/max_psnr', max_psnr, epoch)
            writer.add_scalar('eval/max_ssim', max_ssim, epoch)
        else:
            is_best = False
            
        if epoch % config['save_per_epoch'] == 0 or epoch == total_epochs:
            save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, logger, is_best=is_best)
        
        logger.info(f'Train: [{epoch}/{total_epochs}] Max Valid PSNR: {max_psnr:.4f}, Max Valid SSIM: {max_ssim:.4f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def profile_model(config, profile_forward, model, data_loader, logger):
    with jt.no_grad():
        if profile_forward is not None:
            data_iter = iter(data_loader)
            data = next(data_iter)
            del data_iter
            profile_forward(config, model, data, logger)
        
        n_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"Total Params: {n_parameters:,}")

def train_one_epoch(config, train_forward, model, loss_list, data_loader, optimizer, scaler, epoch, lr_scheduler, writer):
    num_steps = len(data_loader)
    batch_time, data_time, loss_meter, norm_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    losses_meter = [AverageMeter() for _ in range(len(loss_list))]

    print("初始化结束")

    end = time.time()
    for idx, data in enumerate(data_loader):
        # print("进入循环")
        data_time.update(time.time() - end)

        outputs, targets = train_forward(config, model, data)  # 
        losses = loss_list(outputs, targets)
        loss = sum(losses)

        # debug_fuse_grads_once(model, loss)   # 放在 step 前
        optimizer.step(loss) # Jittor核心改动：一步完成反向传播和更新

        # print("一步完成反向传播和更新,loss:", loss.item())

        if not config['train']['lr_scheduler']['t_in_epochs']:
             lr_scheduler.step() # Jittor中，按迭代步进的调度器通常这样调用

        grad_norm = get_grad_norm(model.parameters(), optimizer) # 这里多传入一个优化器，供get_grad_norm使用
        
        batch_size = list(targets.values())[0].size(0)
        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(grad_norm)
        for _loss_meter, _loss in zip(losses_meter, losses):
            _loss_meter.update(_loss.item(), batch_size)
        
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config['print_per_iter'] == 0 or idx == (num_steps -1):
            lr = optimizer.lr
           
            # memory_used = jt.gc.mem_info()["gpu_mem_used"] / 1024 / 1024 # 使用jt.mem_stat()获取当前显存占用
            etas = batch_time.avg * (num_steps - 1 - idx)
            logger.info(
                f'Train: [{epoch}/{config["train"]["epochs"]}][{idx}/{num_steps}]\t'
                f'ETA {datetime.timedelta(seconds=int(etas))} LR {lr:.6f}\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'Loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'
                f'GradNorm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t')
              #   f'Mem {memory_used:.0f}MB')
    # ... (tensorboard logging part)

# def debug_fuse_grads_once(model, loss):
#     if hasattr(model, "_fuse_grad_probed"): 
#         print("偷摸return")
#         return
#     # 取所有 fuse 的 depthwise conv 权重/偏置
#     ws, bs = [], []
#     for f in model.feedback_fuses:
#         # 保证拿到的是 dwconv 里的 depthwise conv（按你的命名就是 [1]）
#         ws.append(f.dwconv[1].weight)
#         bs.append(f.dwconv[1].bias)

#     gws = jt.grad(loss, ws, retain_graph=True)
#     gbs = jt.grad(loss, bs, retain_graph=True)

#     print("\n[Probe] feedback_fuses dwconv grads (mean|abs):")
#     for i,(gw,gb) in enumerate(zip(gws, gbs)):
#         gm = float(gw.abs().mean())
#         bm = float(gb.abs().mean())
#         print(f"  fuse[{i}] dw.weight={gm:.3e}  dw.bias={bm:.3e}")
#     model._fuse_grad_probed = True



def validate(config, test_forward, model, loss_list, data_loader, epoch, writer):
    with jt.no_grad():
        logger.info(f"Valid: [{epoch}/{config['train']['epochs']}]\t")
        batch_time, data_time, loss_meter, psnr_meter, ssim_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        losses_meter = [AverageMeter() for _ in range(len(loss_list))]
        end = time.time()
        for idx, data in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs, targets, img_files, lbl_files = test_forward(config, model, data)

            if config['testset_as_validset']:
                psnr, ssim = test_metric_jittor(config, epoch, outputs[config['test']['which_stage']], targets[config['test']['which_gt']], img_files)
            else:
                psnr, ssim = validate_metric_jittor(config, epoch, outputs[config['test']['which_stage']], targets[config['test']['which_gt']], img_files)

            losses = loss_list(outputs, targets)
            loss = sum(losses)
            batch_size = targets[config['test']['which_gt']].size(0)
            loss_meter.update(loss.item(), batch_size)
            psnr_meter.update(psnr.item(), batch_size)
            ssim_meter.update(ssim.item(), batch_size)
            for _loss_meter, _loss in zip(losses_meter, losses):
                _loss_meter.update(_loss.item(), batch_size)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if config['testset_as_validset'] or idx % config['print_per_iter'] == 0 or idx == (len(data_loader)-1):
               # 哭了，没有在jittor中找到这种类似的函数
               #  memory_used = jt.flags.cuda_max_mem_used / 1024 / 1024
                logger.info(
                    f'Valid: [{epoch}/{config["train"]["epochs"]}][{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'Loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'
                    f'PSNR {psnr_meter.val:.4f} ({psnr_meter.avg:.4f})\t'
                    f'SSIM {ssim_meter.val:.4f} ({ssim_meter.avg:.4f})\t')
                   #  f'Mem {memory_used:.0f}MB\t{os.path.basename(img_files[0])}')
        logger.info(f' * PSNR {psnr_meter.avg:.4f}\tSSIM {ssim_meter.avg:.4f}')
        # ... (tensorboard logging part)
        return psnr_meter.avg, ssim_meter.avg, loss_meter.avg

def validate_metric_jittor(config, epoch, outputs, targets, image_paths):
    with jt.no_grad():
        outputs = jt.clamp(outputs, 0, 1) * 255
        targets = targets * 255
        if config['test']['round']:
            outputs = outputs.round()
            targets = targets.round()
        psnrs = get_psnr_jittor(outputs, targets)
        ssims = get_ssim_jittor(outputs, targets)

        if config['test']['save_image'] and epoch % config['save_per_epoch'] == 0:
            images = jt.concat([outputs, targets], dim=3)
            result_path = os.path.join(config['output'], 'results', f'valid_{epoch:04d}')
            os.makedirs(result_path, exist_ok=True)
            for image, image_path, psnr in zip(images, image_paths, psnrs):
                save_path = os.path.join(result_path, f'{os.path.basename(image_path)[:-4]}_{psnr.item():.2f}.jpg')
                save_image_jittor(image, save_path)

        return psnrs.mean(), ssims.mean()

def test_metric_jittor(config, epoch, outputs, targets, image_paths):
    with jt.no_grad():
        outputs = jt.clamp(outputs, 0, 1) * 255
        targets = jt.clamp(targets, 0, 1) * 255
        if config['test']['round']:
            outputs = outputs.round()
            targets = targets.round()
        psnr = get_psnr_jittor(outputs, targets)
        ssim = get_ssim_jittor(outputs, targets)

        if config['test']['save_image']:
            result_path = os.path.join(config['output'], 'results', f'test_{epoch:04d}')
            os.makedirs(result_path, exist_ok=True)
            save_path = os.path.join(result_path, f'{os.path.basename(image_paths[0])[:-4]}_{psnr.item():.2f}.png')
            save_image_jittor(outputs[0], save_path)

        return psnr, ssim

def throughput(config, forward, model, data_loader, logger):
    with jt.no_grad():
        for idx, data in enumerate(data_loader):
            for i in range(30):
                forward(config, model, data)
            logger.info(f"throughput averaged with 100 times")
            jt.sync_all(True) # Jittor改动：等价于torch.cuda.synchronize()
            tic = time.time()
            for i in range(100):
                pred, label = forward(config, model, data)
            batch_size = list(pred.values())[0].size(0)
            jt.sync_all(True) # Jittor改动：等价于torch.cuda.synchronize()
            toc = time.time()
            logger.info(f"batch_size {batch_size} throughput {(toc - tic) * 1000 / (100 * batch_size)}ms")
            return


if __name__ == '__main__':
    args, config = parse_options()
    phase = 'train' if not args.test else 'test'

    # Jittor改动：设置使用CUDA和CUDNN benchmark
    jt.flags.use_cuda = 1
    jt.cudnn.benchmark = True

    os.makedirs(config['output'], exist_ok=True)
    start_time = time.strftime("%y%m%d-%H%M", time.localtime())
    logger = create_logger(output_dir=config['output'], name=f"{config['tag']}", action=f"{phase}-{start_time}")
    path = os.path.join(config['output'], f"{phase}-{start_time}.yaml")
    
    # ... (git and file saving logic is the same)
    
    copy_cfg(config, path)
    logger.info(f"Full config saved to {path}")
    logger.info("Config:\n" + yaml.dump(ordered_dict_to_dict(config), default_flow_style=False, sort_keys=False))

    # Jittor改动：获取设备信息的方式不同  # 目前不知道为什么这个有问题
    # Jittor改动：使用jt.get_device_info()获取设备信息 ！！！！！
    # device_info = jt.get_device_info()
    # if len(device_info) > 0:
    # # 通过jt.flags.cuda_device_id获取当前使用的GPU索引
    #     current_device_id = jt.flags.cuda_device_id
    #     device_name = device_info[current_device_id]['name']
    #     total_mem = device_info[current_device_id]['mem'] / 1024 / 1024 # 转换为MB
    #     logger.info(f"Current CUDA Device: {device_name}, Total Mem: {int(total_mem)}MB")
    # else:
    #     logger.info("No CUDA device found.")

    logger.info(f"初始化结束，进入main(config)")

    main(config)