# -*- coding: utf-8 -*-

# ***************************************************
# * File        : custom_transforms.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-08
# * Version     : 1.0.080821
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
from typing import Tuple, Union
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange
from torchvision.transforms.functional import (
    resize, 
    InterpolationMode
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DynamicResize(nn.Module):
    """
    Resize so that:
      * the longer side ≤ `max_side_len` **and** is divisible by `patch_size`
      * the shorter side keeps aspect ratio and is also divisible by `patch_size`
    Optionally forbids up-scaling.

    Works on PIL Images, (C, H, W) tensors, or (B, C, H, W) tensors.
    Returns the same type it receives.
    """
    def __init__(self, 
                 patch_size: int, 
                 max_side_len: int, 
                 interpolation: InterpolationMode=InterpolationMode.BICUBIC,
                 allow_upscale: bool=True) -> None:
        super().__init__()
        
        self.patch_size = int(patch_size)
        self.max_side_len = int(max_side_len)
        self.interpolation = interpolation
        self.allow_upscale = allow_upscale
    
    def _get_new_hw(self, h: int, w: int) -> Tuple[int, int]:
        """
        Compute target (h, w) divisible by patch_size.
        """
        long, short = (w, h) if w >= h else (h, w)
        # 1) clamp long side
        target_long = min(self.max_side_len, math.ceil(long / self.patch_size) * self.patch_size)
        if not self.allow_upscale:
            target_long = min(target_long, long)
        # 2) scale factor
        scale = target_long / long
        # 3) compute short side with ceil -> never undershoot
        target_short = math.ceil(short * scale / self.patch_size) * self.patch_size
        target_short = max(target_short, self.patch_size)  # just in case

        return (target_short, target_long) if w >= h else (target_long, target_short)

    def forward(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            w, h = img.size
            new_h, new_w = self._get_new_hw(h, w)
            return resize(img, [new_h, new_w], interpolation=self.interpolation)

        if not torch.is_tensor(img):
            raise TypeError(
                "DynamicResize expects a PIL Image or a torch.Tensor; "
                f"got {type(img)}"
            )

        # tensor path
        batched = img.ndim == 4
        if img.ndim not in (3, 4):
            raise ValueError(
                "Tensor input must have shape (C,H,W) or (B,C,H,W); "
                f"got {img.shape}"
            )

        # operate batch-wise
        imgs = img if batched else img.unsqueeze(0)
        _, _, h, w = imgs.shape
        new_h, new_w = self._get_new_hw(h, w)
        out = resize(imgs, [new_h, new_w], interpolation=self.interpolation)

        return out if batched else out.squeeze(0)


class SplitImage(nn.Module):
    """
    Split (B, C, H, W) image tensor into square patches.

    Returns:
        patches: (B·n_h·n_w, C, patch_size, patch_size)
        grid:    (n_h, n_w)  - number of patches along H and W
    """
    def __init__(self, patch_size: int) -> None:
        super().__init__()

        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim == 3:  # add batch dim if missing
            x = x.unsqueeze(0)

        b, c, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f'Image size {(h,w)} not divisible by patch_size {self.patch_size}')

        n_h, n_w = h // self.patch_size, w // self.patch_size
        patches = rearrange(
            x, 
            pattern='b c (nh ph) (nw pw) -> (b nh nw) c ph pw',
            ph=self.patch_size, 
            pw=self.patch_size,
        )
        
        return patches, (n_h, n_w)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
