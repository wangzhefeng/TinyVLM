# -*- coding: utf-8 -*-

# ***************************************************
# * File        : vision_transformer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070815
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
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ViTPatchEmbeddings(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
    """
    
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        pass


class ViTMultiHeadAttention(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
    https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
    """
    
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        pass


class ViTMLP(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
    """
    
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        pass


class ViTBlock(nn.Module):
    """
    https://github.com/karpathy/nanoGPT/blob/master/model.py#L94
    """
    
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        pass


class ViT(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        pass

    @classmethod
    def from_pretrained(cls, cfg):
        """
        Load the model from a pretrained HuggingFace model 
        (we don't want to have to train the Vision Backbone from scratch)
        """
        pass




# 测试代码 main 函数
def main():
    from utils.log_util import logger

if __name__ == "__main__":
    main()
