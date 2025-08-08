# -*- coding: utf-8 -*-

# ***************************************************
# * File        : collators.py
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
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class BaseCollator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [
            F.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id)
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            F.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id)
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            F.pad(attention_mask, (max_length - len(attention_mask), 0), value=0)
            for attention_mask in batch["attention_mask"]
        ]
    
    def prepare_batch(self, batch, max_length=None):
        """
        batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        """
        # convert batch to a dict of list of tensors 
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)
        # Pad samples to max_length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        # dictionaries in Python are mutable and passed by reference
        self._pad_batch(batch, max_len)
        
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }
    
    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(
                batch["input_ids"], 
                batch["labels"], 
                batch["attention_mask"], 
                batch["images"]
            )
            if len(ids) <= max_length
        ]
        if not filtered:
            return [], [], [], []
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        
        return {
            "input_ids": list(batch_token_ids),
            "labels": list(batch_labels),
            "attention_mask": list(batch_attentions),
            "images": list(batch_images)
        }


class VQACollator(BaseCollator):
    """
    Visual Question Answering Collator
    """

    def __init__(self, tokenizer, max_length):
        super().__init__(tokenizer)

        self.max_length = max_length
    
    def _pad_batch(self, batch, max_length):
        """
        Reimplementing to use `-100` as the pad value for labels, 
        so that it's ignored by the loss
        """
        batch["input_ids"] = [
            F.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id)
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            F.pad(labels, (max_length - len(labels), 0), value=-100)
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            F.pad(attention_mask, (max_length - len(attention_mask), 0), value=0)
            for attention_mask in batch["attention_mask"]
        ]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)

        return batch




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
