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

import torch
from torch.utils.data import DataLoader
from datasets import (
    get_dataset_config_names,
    load_dataset, 
    concatenate_datasets,
)
from torch.utils.data import DistributedSampler

from nanoVLM.data_provider.processors import (
    get_image_processor, 
    get_tokenizer,
)
from nanoVLM.utils_vlm import (
    seed_worker,
    is_master, is_dist, 
    get_world_size, get_rank,
)
from nanoVLM.data_provider.collators import VQACollator
from nanoVLM.data_provider.dataset import VQADataset
from nanoVLM.data_provider.advanced_datasets import ConstantLengthDataset
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def get_dataloaders(train_cfg, vlm_cfg):
    # ------------------------------
    # image processor
    # ------------------------------
    image_processor = get_image_processor(
        max_img_size=vlm_cfg.max_img_size, 
        splitted_image_size=vlm_cfg.vit_img_size,
    )
    # ------------------------------
    # tokenizer
    # ------------------------------
    tokenizer = get_tokenizer(
        name=vlm_cfg.lm_tokenizer, 
        extra_special_tokens=vlm_cfg.vlm_extra_tokens, 
        chat_template=vlm_cfg.lm_chat_template,
        cache_dir=vlm_cfg.model_cache_dir,
    )
    logger.info(f"tokenizer: \n{tokenizer}")
    # ------------------------------
    # Load and combine all training datasets
    # ------------------------------
    combined_train_data = []
    # dataset name
    if "all" in train_cfg.train_dataset_name:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)
    else:
        dataset_names_to_load = train_cfg.train_dataset_name
    # logger.info(f"debug::dataset_names_to_load: {dataset_names_to_load}")
    # dataset load
    for dataset_name in dataset_names_to_load:
        try:
            train_ds = load_dataset(
                path=train_cfg.train_dataset_path, 
                name=dataset_name, 
                cache_dir=train_cfg.train_dataset_cache_dir
            )
            train_ds["train"][0]  # Check if the dataset is loaded correctly
            combined_train_data.append(train_ds["train"])
        except Exception as e:
            if is_master():
                logger.info(f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
            continue
    # logger.info(f"debug::combined_train_data: \n{combined_train_data[0]}")
    # ------------------------------
    # train dataset
    # ------------------------------
    if not combined_train_data:
        raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")
    train_ds = concatenate_datasets(combined_train_data)

    # Shuffle the training dataset, so train and valid get equal contributions from all concatenated datasets
    train_ds = train_ds.shuffle(seed=0)
    
    # Shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
    if is_dist():
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())
    
    # logger.info(f"debug::train_ds: \n{train_ds}")
    # ------------------------------
    # train dataset split
    # ------------------------------
    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)
    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size
    logger.info(f"train_size: {train_size}")
    logger.info(f"val_size: {val_size}")
    # ------------------------------
    # Create Dataset
    # ------------------------------
    train_dataset = VQADataset(
        train_ds.select(range(0, train_size)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )
    train_dataset = ConstantLengthDataset(
        train_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length, 
        num_of_sequences=train_cfg.batch_size*64, 
        queue_size=train_cfg.batch_size*64*2,
        max_images_per_example=train_cfg.max_images_per_example, 
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )
    val_dataset = VQADataset(
        train_ds.select(range(train_size, total_samples)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )
    
    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    
    # 设置生成随机数的种子
    g = torch.Generator()
    g.manual_seed(0)
    # ------------------------------
    # Create dataloaders
    # ------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,  # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        # shuffle=True,
        generator=g,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False,  # Usually False for validation
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        # shuffle=False,
        generator=g,
    )

    return train_loader, val_loader




# 测试代码 main 函数
def main():
    from utils.log_util import logger
    from nanoVLM.models.config import VLMConfig, TrainConfig
    
    # config
    train_cfg = TrainConfig()
    vlm_cfg = VLMConfig()
    
    # dataloader
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)
    for batch in train_loader:
        logger.info(f"debug::batch: \n{batch}")
        break

if __name__ == "__main__":
    main()
