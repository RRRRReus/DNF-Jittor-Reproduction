# 文件: utils/metrics.py (Jittor完整版)

import warnings
import jittor as jt
from jittor import nn
import jittor.nn as F # Jittor改动：使用F作为jittor.nn的别名，方便替换

class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, x, gt):
        # Jittor改动: torch.mean/log10/sqrt -> jt.mean/log10/sqrt
        mse = jt.mean((x - gt) ** 2, dims=[1, 2, 3])
        return 20 * (jt.log(jt.array(255.0) / jt.sqrt(mse)) / jt.log(jt.array(10.0)))

def get_ssim_jittor(x, gt):
    # Jittor改动：调用我们下面翻译好的ssim函数
    return ssim(x, gt, size_average=False)

def get_psnr_jittor(x, gt, data_range=255.0):
    # Jittor改动: torch.mean/log10/sqrt -> jt.mean/log10/sqrt
    mse = jt.mean((x - gt) ** 2, dims=[1, 2, 3])

    # 查阅文档，发现 log2 运算在 misc 模块下面，没有log10
    return 20 * (jt.log(data_range / jt.sqrt(mse)) / jt.log(jt.array(10.0)))

#
# --- 以下是SSIM和MS-SSIM的完整实现 ---
#

def _fspecial_gauss_1d(size, sigma):
    # Jittor改动: torch.arange/exp -> jt.arange/exp
    coords = jt.arange(size, dtype=jt.float32)
    coords -= size // 2

    g = jt.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    # Jittor改动: F.conv2d/conv3d 来自 jittor.nn
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            # Jittor改动: .transpose 在Jittor中同样可用
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )
    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    K1, K2 = K
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Jittor改动: 无需手动.to(device, dtype)，Jittor自动处理
    
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    
    # Jittor改动: torch.flatten -> jt.flatten
    ssim_per_channel = jt.flatten(ssim_map, 2).mean(-1)
    cs = jt.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, K=(0.01, 0.03), nonnegative_ssim=False):
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    # for d in range(len(X.shape) - 1, 1, -1):
    #     # Jittor改动: .squeeze(dim=d) -> .squeeze(d)
    #     X = X.squeeze(d)
    #     Y = Y.squeeze(d)

    # Jittor改动：在squeeze前增加维度检查
    # 这个while循环会从后往前检查，只有当最后一个维度尺寸为1时，才执行squeeze
    while len(X.shape) > 4 and X.shape[-1] == 1:
        X = X.squeeze(len(X.shape) - 1)
        Y = Y.squeeze(len(Y.shape) - 1) 


    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")
    
    # Jittor改动: .type() -> .dtype
    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        # Jittor改动: .repeat 在Jittor中同样可用
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        # Jittor改动: torch.relu -> jt.relu
        ssim_per_channel = jt.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)):
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(d)
        Y = Y.squeeze(d)
    
    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), f"Image size should be larger than {(win_size - 1) * (2 ** 4)}"

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    # Jittor改动: torch.tensor -> jt.array, 无需device和dtype
    weights = jt.array(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            # Jittor改动: torch.relu -> jt.relu
            mcs.append(jt.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = jt.relu(ssim_per_channel)
    # Jittor改动: torch.stack -> jt.stack
    mcs_and_ssim = jt.stack(mcs + [ssim_per_channel], dim=0)
    # Jittor改动: torch.prod -> jt.prod, .view -> .reshape
    ms_ssim_val = jt.prod(mcs_and_ssim ** weights.reshape((-1, 1, 1)), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)

class SSIM(nn.Module):
    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, K=(0.01, 0.03), nonnegative_ssim=False):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y, data_range=self.data_range, size_average=self.size_average, win=self.win, K=self.K, nonnegative_ssim=self.nonnegative_ssim)

class MS_SSIM(nn.Module):
    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, weights=None, K=(0.01, 0.03)):
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(X, Y, data_range=self.data_range, size_average=self.size_average, win=self.win, weights=self.weights, K=self.K)