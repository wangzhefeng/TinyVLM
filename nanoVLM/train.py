# -*- coding: utf-8 -*-

# ***************************************************
# * File        : nanoVLM.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070814
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
from dataclasses import asdict
import warnings
warnings.filterwarnings("ignore")

import torch
import wandb

from nanoVLM.models import config
from nanoVLM.models.utils import (
    is_master, is_dist, 
    init_dist, destory_dist,
    seed_torch, seed_worker,
    wrap_model,
    get_run_name, get_world_size,
)
# set torch seed
seed_torch()
# dataset
from nanoVLM.data_provider.data_factory import get_dataloaders
from nanoVLM.data_provider.processors import get_tokenizer
from nanoVLM.models.vision_language_model import VisionLanguageModel

# Otherwise, the tokenizer will throw a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for "Decompressed data too large" error with certain PNGs
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CUNK = 100 * 1024 * 1024

# local log
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--compile', type=bool, help='Use torch.compile to optimize the model')
    parser.add_argument('--log_wandb', type=bool, help='Log to wandb')
    parser.add_argument('--no_log_wandb', action='store_true', help='Do not log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')

    args = parser.parse_args()
    return args


def train(train_cfg, vlm_cfg):
    # data loader
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)
    # tokenizer
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    # run name
    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    if train_cfg.log_wandb and is_master():
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
    logger.info(f"Run name: {run_name}")
    # wandb run
    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM",
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg), 
            },
            name=run_name,
        )
    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    # Model training summary
    if is_master():
        logger.info(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        logger.info(f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            logger.info(f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}")
        
        logger.info(f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            logger.info(f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, 
    # but a newly initialized modality projection layer, 
    # it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, 
    # but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = [{"params": list(model.MP.parameters()), "lr": train_cfg.lr_mp}]
    if train_cfg.lr_backbones > 0:
        param_groups.append({
            "params": list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 
            "lr": train_cfg.lr_backbones
        })
    else:
        for p in list(model.decoder.parameters()) + list(model.vision_encoder.parameters()):
            p.requires_grad = False
    # optimizer
    optimizer = torch.optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    # device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()
    logger.info(f"Using device: {device}")

    # model to device
    model.to(device)

    # model compile
    if train_cfg.compile:
        model = torch.compile(model)
    
    # distributed training
    if is_dist():
        model = wrap_model(model)
    # ------------------------------
    # model training initialization
    # ------------------------------
    epoch_times = []
    best_val_loss = float("inf")
    global_step = 0
    epoch = 0
    # training stats accumulators 
    accumulated_stats = {
        "token_per_second": [],
        "data_load_time": [],
        "fw_bw_time": [],
        "post_process_time": [],
        "images_per_sample": [],
    }
    while global_step < train_cfg.max_training_steps:
        pass




# 测试代码 main 函数
def main():
    # args
    args = parse_args()
    # model configs
    vlm_cfg = config.VLMConfig()
    # model training configs
    train_cfg = config.TrainConfig()
    # update configs
    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.no_log_wandb is True:
        train_cfg.log_wandb = False
    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False
    
    # initiation distributed training
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
    
    # print vlm_cfg and train_cfg
    if is_master():
        logger.info("--- VLM Config ---")
        logger.info(vlm_cfg)
        logger.info("--- Train Config ---")
        logger.info(train_cfg) 
    
    # model training
    # train(train_cfg, vlm_cfg)

    # destory distributed training
    if is_dist():
        destory_dist()

if __name__ == "__main__":
    main()
