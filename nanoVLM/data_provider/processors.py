# -*- coding: utf-8 -*-

# ***************************************************
# * File        : processors.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-05
# * Version     : 1.0.080522
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer
import torchvision.transforms as transforms

from nanoVLM.data.custom_transforms import DynamicResize, SplitImage

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    TOKENIZERS_CACHE = {}
    if name not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens

        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_init_kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        # update TOKENIZERS_CACHE
        TOKENIZERS_CACHE[name] = tokenizer
    
    return TOKENIZERS_CACHE[name]


def get_image_processor(max_img_size: int, splitted_image_size: int):
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size),
        transforms.ToTensor(),
        SplitImage(splitted_image_size),
    ])


def get_image_string(tokenizer, splitted_image_counts: List, mp_image_token_length: int):
    image_string = ""
    # splitted_image_counts is a list of tuple (n_h, n_w)
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f"r{i+1}c{j+1}")
                image_string += tokenizer.image_token * mp_image_token_length
    
    return image_string




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
