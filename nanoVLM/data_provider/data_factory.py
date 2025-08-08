# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-08
# * Version     : 1.0.080823
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

from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

from nanoVLM.data_provider.processors import (
    get_image_processor, 
    get_tokenizer,
)
from nanoVLM.data_provider.collators import VQACollator
from nanoVLM.data_provider.dataset import VQADataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    image_processor = get_image_processor(max_img_size=vlm_cfg.vit_img_size)
    # tokenizer
    tokenizer = get_tokenizer(
        name=vlm_cfg.lm_tokenizer, 
        extra_special_tokens=vlm_cfg.vlm_extra_tokens, 
        chat_template=vlm_cfg.lm_chat_template
    )
    # Load and combine all training datasets
    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(path=train_cfg.train_dataset_path, name=dataset_name)
        combined_train_data.append(train_ds["train"])
    train_ds = concatenate_datasets(combined_train_data)
    # Shuffle the training dataset, so train and valid get equal contributions from all concatenated datasets
    train_ds = train_ds.shuffle(seed=0)
    
    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)
    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size
    
    # Create Dataset
    train_dataset = VQADataset(
        train_ds.select(range(0, train_size)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )
    val_dataset = VQADataset(
        train_ds.select(range(train_size, total_samples)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )
    
    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=vqa_collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=vqa_collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader




# 测试代码 main 函数
def main():
    from utils.log_util import logger

if __name__ == "__main__":
    main()
