# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-14
# * Version     : 1.0.081423
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
import warnings
warnings.filterwarnings("ignore")

import requests
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# device
device = device_setting(verbose=True)

# processor
processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    cache_dir="./downloaded_models",
)

# model
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    cache_dir="./downloaded_models"
)
model.to(device)

# image
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# prompt
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

# inputs
inputs = processor(prompt, image, return_tensors="pt").to(device)

# output
output = model.generate(**inputs, max_new_tokens=100)

# output decoded
output_decoded = processor.decode(output[0], skip_special_tokens=True)

logger.info(f"output[0]: {output_decoded}")



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
