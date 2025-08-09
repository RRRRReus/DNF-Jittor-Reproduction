#!/usr/bin/env python

import os
import time

import imageio
import jittor as jt  # Jittor：将 torch 替换为 jittor
import numpy as np
import rawpy
import tqdm
from jittor.dataset import Dataset  # Jittor：替换 torch.utils.data.Dataset
from utils.meters import AverageMeter

from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MCRDictSet(Dataset):  # Jittor：继承自 jittor.dataset.Dataset

    def __init__(self, data_path, image_list_file, patch_size=None, split='train', load_npy=True, repeat=1,
                 raw_ext='ARW', max_samples=None, max_clip=1.0, min_clip=None, only_00=False,
                 transpose=True, h_flip=True, v_flip=True, rotation=False, ratio=True, 
                 # Jittor改动：从这里开始，添加DataLoader会传递过来的所有参数
                 batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, **kwargs):
        super().__init__()  # Jittor：好习惯是调用 super().__init__()
        """
        :param data_path: 数据集目录
        :param image_list_file: 包含图像文件名的列表文件
        :param patch_size: 如果为None，返回完整图像，否则返回图像块
        :param split: 'train' 或 'valid'
        :param upper: 用于调试的最大图像数量
        """
        assert os.path.exists(data_path), "data_path: {} not found.".format(data_path)
        self.data_path = data_path
        image_list_file = os.path.join(data_path, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        self.load_npy = load_npy
        self.raw_ext = raw_ext
        self.max_clip = max_clip
        self.min_clip = min_clip
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotation = rotation
        self.ratio = ratio
        self.only_00 = only_00
        self.repeat = repeat

        # Jittor改动：将这些参数保存为类的属性
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # 以下 AverageMeter 部分保持不变
        self.raw_short_read_time = AverageMeter()
        self.raw_short_pack_time = AverageMeter()
        self.raw_short_post_time = AverageMeter()
        self.raw_long_read_time = AverageMeter()
        self.raw_long_pack_time = AverageMeter()
        self.raw_long_post_time = AverageMeter()
        self.npy_long_read_time = AverageMeter()
        self.data_aug_time = AverageMeter()
        self.data_norm_time = AverageMeter()
        self.count = 0

        self.block_size = 2
        self.black_level = 0
        self.white_level = 255

        # 以下文件预加载部分保持不变
        self.raw_input_path = []
        self.raw_gt_path = []
        self.rgb_gt_path = []
        self.rgb_gt_dict = {}
        self.raw_input_list = []
        self.raw_gt_dict = {}
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                raw_input_path, raw_gt_path, rgb_gt_path = img_pair.strip().split(' ')
                self.raw_input_path.append(os.path.join(self.data_path, raw_input_path))
                self.raw_gt_path.append(os.path.join(self.data_path, raw_gt_path))
                self.rgb_gt_path.append(os.path.join(self.data_path, rgb_gt_path))
                raw_input = imageio.imread(os.path.join(self.data_path, raw_input_path))
                self.raw_input_list.append(raw_input)

                raw_gt = imageio.imread(os.path.join(self.data_path, raw_gt_path))
                raw_gt_name = os.path.basename(raw_gt_path)
                if raw_gt_name not in self.raw_gt_dict:
                    self.raw_gt_dict[raw_gt_name] = raw_gt

                rgb_gt = imageio.imread(os.path.join(self.data_path, rgb_gt_path)).transpose(2, 0, 1)
                rgb_gt_name = os.path.basename(rgb_gt_path)
                if rgb_gt_name not in self.rgb_gt_dict:
                    self.rgb_gt_dict[rgb_gt_name] = rgb_gt

                if max_samples and i == max_samples - 1:
                    break

        print("processing: {} images for {}".format(len(self.raw_input_path), self.split))
        self.set_attrs(total_len=len(self.raw_input_path) * self.repeat) # Jittor：为 DataLoader 设置数据集总长度

    # Jittor: 在Jittor数据集中，__len__ 方法由 self.set_attrs(total_len=...) 代替
    # def __len__(self):
    #     return len(self.raw_input_path) * self.repeat

    def print_time(self):
        # 此函数保持不变
        print('self.raw_short_read_time:', self.raw_short_read_time.avg)
        # ... (其他打印)

    def __getitem__(self, index):
        self.count += 1
        idx = index // self.repeat
        if self.count % 100 == 0 and False:
            self.print_time()
        info = self.raw_input_path[idx]
        img_file = info

        start = time.time()
        noisy_raw = self.raw_input_list[idx]
        if self.patch_size is None:
            noisy_raw = self._pack_raw(noisy_raw)
        self.raw_short_read_time.update(time.time() - start)

        lbl_file = self.rgb_gt_path[idx]
        start = time.time()
        clean_rgb = self.rgb_gt_dict[os.path.basename(self.rgb_gt_path[idx])]
        self.raw_long_post_time.update(time.time() - start)

        start = time.time()
        clean_raw = self.raw_gt_dict[os.path.basename(self.raw_gt_path[idx])]
        if self.patch_size is None:
            clean_raw = self._pack_raw(clean_raw)
        self.raw_long_read_time.update(time.time() - start)

        if self.patch_size:
            start = time.time()
            patch_size = self.patch_size
            H, W = clean_rgb.shape[1:3]
            if self.split == 'train':
                # Jittor：将 torch.randint 替换为 numpy.random.randint，用于在CPU上生成随机数
                yy = np.random.randint(0, (H - patch_size) // self.block_size) if (H - patch_size) > 0 else 0
                xx = np.random.randint(0, (W - patch_size) // self.block_size) if (W - patch_size) > 0 else 0
            else:
                yy, xx = (H - patch_size) // self.block_size // 2, (W - patch_size) // self.block_size // 2
            input_patch = self._pack_raw(noisy_raw, yy, xx)
            clean_raw_patch = self._pack_raw(clean_raw, yy, xx)
            gt_patch = clean_rgb[:, yy * self.block_size:yy * self.block_size + patch_size,
                       xx * self.block_size:xx * self.block_size + patch_size]

            # Jittor：将 torch.randint 替换为 numpy.random.randint
            if self.h_flip and np.random.randint(0, 2) == 1 and self.split == 'train':  # 随机水平翻转
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
                clean_raw_patch = np.flip(clean_raw_patch, axis=2)
            if self.v_flip and np.random.randint(0, 2) == 1 and self.split == 'train':  # 随机垂直翻转
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
                clean_raw_patch = np.flip(clean_raw_patch, axis=1)
            if self.transpose and np.random.randint(0, 2) == 1 and self.split == 'train':  # 随机转置
                input_patch = np.transpose(input_patch, (0, 2, 1))
                gt_patch = np.transpose(gt_patch, (0, 2, 1))
                clean_raw_patch = np.transpose(clean_raw_patch, (0, 2, 1))
            if self.rotation and self.split == 'train':
                raise NotImplementedError('rotation')

            noisy_raw = input_patch.copy()
            clean_rgb = gt_patch.copy()
            clean_raw = clean_raw_patch.copy()
            self.data_aug_time.update(time.time() - start)

        start = time.time()
        noisy_raw = (np.float32(noisy_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        clean_raw = (np.float32(clean_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        clean_rgb = np.float32(clean_rgb) / np.float32(255)
        self.data_norm_time.update(time.time() - start)

        img_num = int(self.raw_input_path[idx][-23:-20])
        img_expo = int(self.raw_input_path[idx][-8:-4], 16)

        if img_num < 500:
            gt_expo = 12287
        else:
            gt_expo = 1023
        ratio = gt_expo / img_expo

        if self.ratio:
            noisy_raw = noisy_raw * ratio
        if self.max_clip is not None:
            noisy_raw = np.minimum(noisy_raw, self.max_clip)
        if self.min_clip is not None:
            noisy_raw = np.maximum(noisy_raw, self.min_clip)

        clean_rgb = clean_rgb.clip(0.0, 1.0)
        
        # Jittor：将 torch.from_numpy(...).float() 替换为更直接的 jt.array(...)
        # 不知为何这个地方出现了及其诡异的错误
        if not isinstance(noisy_raw, jt.Var):
            final_noisy_raw = np.ascontiguousarray(noisy_raw)
            noisy_raw = jt.array(final_noisy_raw)
        else:
            print(f">>> [WARNING] noisy_raw was already a Jittor Var!")

    # 检查 clean_rgb
        if not isinstance(clean_rgb, jt.Var):
            final_clean_rgb = np.ascontiguousarray(clean_rgb)
            clean_rgb = jt.array(final_clean_rgb)
        else:
            print(f">>> [WARNING] clean_rgb was already a Jittor Var!") # 我们怀疑问题在这里

    # 检查 clean_raw
        if not isinstance(clean_raw, jt.Var):
            final_clean_raw = np.ascontiguousarray(clean_raw)
            clean_raw = jt.array(final_clean_raw)
        else:
            print(f">>> [WARNING] clean_raw was already a Jittor Var!")
        
        # Jittor：返回一个元组(tuple)通常更常见，但字典也可以
        return {
            'noisy_raw': noisy_raw,
            'clean_raw': clean_raw,
            'clean_rgb': clean_rgb,
            'img_file': img_file,
            'lbl_file': lbl_file,
            'img_expo': img_expo,
            'gt_expo': gt_expo,
            'ratio': ratio
        }

    def _pack_raw(self, raw, hh=None, ww=None):
        # 这个函数完全使用numpy，保持不变
        if self.patch_size is None:
            assert hh is None and ww is None
        H, W = raw.shape
        im = np.expand_dims(raw, axis=0)
        if self.patch_size is None:
            out = np.concatenate((im[:, 0:H:2, 0:W:2],
                                  im[:, 0:H:2, 1:W:2],
                                  im[:, 1:H:2, 1:W:2],
                                  im[:, 1:H:2, 0:W:2]), axis=0)
        else:
            h1 = hh * 2
            h2 = hh * 2 + self.patch_size
            w1 = ww * 2
            w2 = ww * 2 + self.patch_size
            out = np.concatenate((im[:, h1:h2:2, w1:w2:2],
                                  im[:, h1:h2:2, w1 + 1:w2:2],
                                  im[:, h1 + 1:h2:2, w1 + 1:w2:2],
                                  im[:, h1 + 1:h2:2, w1:w2:2]), axis=0)
        return out