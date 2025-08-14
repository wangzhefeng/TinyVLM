# -*- coding: utf-8 -*-

# ***************************************************
# * File        : finetuning.py
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

import torch
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    TrainingArguments, 
    LlavaForConditionalGeneration
)
from trl.cli import TrlParser
from datasets import load_dataset
from trl import SFTTrainer, SFT

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# command args
parser = TrlParser(SftScriptArguments, TrainingArguments)
args, training_args = parser.parse_args_and_config()

# chat template
LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""

# model id
model_id = "llava-hf/llava-1.5-7b-hf"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = LLAVA_CHAT_TEMPLATE

# processor
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

# model
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)


class LLavaDataCollator:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

data_collator = LLavaDataCollator(processor)


# dataset
raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

# model training
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text", # need a dummy field
    tokenizer=tokenizer,
    data_collator=data_collator,
    dataset_kwargs={"skip_prepare_dataset": True},
)
trainer.train()


# model save
trainer.save_model(training_args.output_dir)
trainer.push_to_hub()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
