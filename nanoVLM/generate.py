# -*- coding: utf-8 -*-

# ***************************************************
# * File        : generate.py
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
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image

from nanoVLM.models.utils import seed_torch
seed_torch(0)
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from an image with nanoVLM")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF.")
    parser.add_argument("--hf_model", type=str, default="lusxvr/nanoVLM-450M",
                        help="HuggingFace repo ID to download from incase --checkpoint isnt set.")
    parser.add_argument("--image", type=str, default="assets/image.png",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum number of tokens per output")
    
    args = parser.parse_args()
    return args





# 测试代码 main 函数
def main():
    # args
    args = parse_args()
    logger.info(f"args: {args}")

    # device
    device = device_setting(verbose=True)

    # model load
    source = args.checkpoint if args.checkpoint else args.hf_model
    logger.info(f"Loading model weights from: {source}")
    model = None
    model.eval()

    # tokenizer
    tokenizer = None

if __name__ == "__main__":
    main()
