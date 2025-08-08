# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-09
# * Version     : 1.0.080900
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def synchronized_dataloader_step(train_loader, is_dist):
    """
    Create a synchronized iterator that handles uneven data distribution in DDP.
    All ranks will stop when the first rank runs out of data.
    This happens because when packing a presharded dataset, a rank might have less groups than the others.
    """
    if not is_dist:
        # For single GPU, we don't need synchronization.
        for batch in train_loader:
            yield batch
        return
    
    # For DDP, we need synchronization.
    train_iter = iter(train_loader)
    
    while True:
        try:
            batch = next(train_iter)
            has_data = torch.tensor(1, device=torch.cuda.current_device())
        except StopIteration:
            batch = None
            has_data = torch.tensor(0, device=torch.cuda.current_device())
        
        # We synchronize across all ranks. If any rank is out of data, all ranks stop.
        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
        
        if has_data.item() == 0:
            # At least one rank is out of data. All ranks should stop.
            break
        yield batch
    
    return None




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
