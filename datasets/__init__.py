# 文件: datasets/__init__.py (Jittor版本)

import importlib
from copy import deepcopy
from os import path as osp
from glob import glob
# Jittor改动: 导入jittor
import jittor as jt

from utils.registry import DATASET_REGISTRY

__all__ = ['build_train_loader', 'build_valid_loader', 'build_test_loader']

# 这部分自动扫描和导入数据集模块的代码是纯Python，无需改动
dataset_folder = osp.dirname(osp.abspath(__file__))
# 修正一个潜在bug：确保只匹配.py文件，避免匹配到.pyc等文件
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in glob(osp.join(dataset_folder, '*_dataset.py'))]
_dataset_modules = [importlib.import_module(f'datasets.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_cfg, split: str):
    # Jittor改动说明：这个函数负责解析配置并从注册器获取Dataset类，与框架无关，无需改动
    assert split.upper() in ['TRAIN', 'VALID', 'TEST']

    # 为了安全深拷贝一下
    dataset_cfg = deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop('type')
    process_cfg = dataset_cfg.pop('process')
    split_cfg = dataset_cfg.pop(split)

    # 在数据集注册器里找对应的dataset
    dataset = DATASET_REGISTRY.get(dataset_type)(
        **dataset_cfg,
        **process_cfg,
        **split_cfg,
        split=split
    )
    print("build_dataset结束")
    return dataset

def build_train_loader(dataset_cfg):
    train_dataset = build_dataset(dataset_cfg, 'train')
    # Jittor改动: 将 torch.utils.data.DataLoader 替换为 jittor.dataset.DataLoader
    # Jittor改动: Jittor的DataLoader没有persistent_workers参数，将其移除
    train_dataloader = jt.dataset.DataLoader(
        train_dataset, 
        batch_size=dataset_cfg['train']['batch_size'],
        shuffle=True, 
        num_workers=dataset_cfg['num_workers'],
        pin_memory=dataset_cfg['pin_memory']
    )
    print("train_loader构建完成")
    return train_dataloader

def build_valid_loader(dataset_cfg, num_workers=None):
    valid_dataset = build_dataset(dataset_cfg, 'valid')
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']
    # Jittor改动: 将 torch.utils.data.DataLoader 替换为 jittor.dataset.DataLoader
    # Jittor改动: Jittor的DataLoader没有persistent_workers参数，将其移除
    valid_dataloader = jt.dataset.DataLoader(
        valid_dataset, 
        batch_size=dataset_cfg['valid']['batch_size'],
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=dataset_cfg['pin_memory']
    )
    return valid_dataloader


def build_test_loader(dataset_cfg, num_workers=None):
    test_dataset = build_dataset(dataset_cfg, 'test')
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']
    # Jittor改动: 将 torch.utils.data.DataLoader 替换为 jittor.dataset.DataLoader
    # Jittor改动: Jittor的DataLoader没有persistent_workers参数，将其移除
    test_dataloader = jt.dataset.DataLoader(
        test_dataset, 
        batch_size=dataset_cfg['test']['batch_size'],
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=dataset_cfg['pin_memory']
    )
    return test_dataloader