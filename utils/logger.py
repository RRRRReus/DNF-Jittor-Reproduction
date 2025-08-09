# 文件: utils/logger.py (Jittor版本，与原版相同)

# --- 说明 ---
# 这个文件只依赖Python标准库(os, sys, logging, functools)
# 和通用的第三方库(termcolor)，与深度学习框架(PyTorch/Jittor)无关。
# 因此，它无需任何修改即可在Jittor项目中使用。

import os
import sys
import logging
import functools
from termcolor import colored

logger = None

@functools.lru_cache()
def create_logger(output_dir, dist_rank=None, name='', action='train'):
    global logger
    if logger is not None:
        return logger
    # create logger
    logger = logging.getLogger(name) # 稍微优化，使用传入的name
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # Jittor改动说明：Jittor同样支持分布式训练，dist_rank的逻辑可以保留
    # 您需要确保在调用create_logger时，传入的dist_rank来自于Jittor的分布式环境
    if dist_rank is not None:
        # create console handlers for master process
        if dist_rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(output_dir, f'{action}-rank-{dist_rank}.log'), mode='a')
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(os.path.join(output_dir, f'{action}.log'), mode='a')
        
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger