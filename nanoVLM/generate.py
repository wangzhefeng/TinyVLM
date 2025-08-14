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

# utils
from nanoVLM.utils_vlm import seed_torch
from utils.device import device_setting
# set torch seed
seed_torch(0)
# dataset
from nanoVLM.data_provider.processors import get_tokenizer, get_image_processor
# model
from nanoVLM.models.vision_language_model import VisionLanguageModel

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

    # model path
    source = args.checkpoint if args.checkpoint else args.hf_model
    logger.info(f"Loading model weights from: {source}")
    
    # model
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    # tokenizer
    tokenizer = get_tokenizer(name=model.cfg.lm_tokenizer, extra_special_tokens=model.cfg.vlm_extra_tokens)

    # image processor
    image_processor = get_image_processor(max_img_size=model.cfg.max_img_size, splitted_image_size=model.cfg.vit_img_size)
    img = Image.open(args.image).convert("RGB")
    processed_image, splitted_image_count = image_processor(img)
    vit_patch_size = splitted_image_count[0] * splitted_image_count[1]

    # prompt
    messages = [{
        "role": "user",
        "content": tokenizer.image_token * model.cfg.mp_image_token_length * vit_patch_size + args.prompt
    }]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt).to(device)
    img_t = processed_image.to(device)

    logger.info(f"\nInput:\n{args.prompt}\n\nOutputs:")
    for i in range(args.generations):
        gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        logger.info(f"  >> Generation {i+1}: {out}")

if __name__ == "__main__":
    main()
