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
import time
import argparse
import contextlib
import statistics
from dataclasses import asdict
import warnings
warnings.filterwarnings("ignore")

import torch
import wandb

# config
from nanoVLM.models import config
# utils
from nanoVLM.utils import (
    is_master, is_dist, 
    init_dist, destory_dist,
    seed_torch, seed_worker,
    wrap_model, dist_gather,
    get_run_name, get_world_size, get_lr,
)
# set torch seed
seed_torch()
# dataset
from nanoVLM.data_provider.data_factory import get_dataloaders
from nanoVLM.data_provider.processors import get_tokenizer
from nanoVLM.data_provider.data_utils import synchronized_dataloader_step
# model
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


# TODO
def valid(train_cfg, vlm_cfg, model, ):
    pass


def train(train_cfg, vlm_cfg):
    # ------------------------------
    # TODO data loader
    # ------------------------------
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)
    # ------------------------------
    # TODO tokenizer
    # ------------------------------
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    # ------------------------------
    # run name
    # ------------------------------
    # wandb run name
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
    # ------------------------------
    # Initialize model
    # ------------------------------
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    # ------------------------------
    # Model training summary
    # ------------------------------
    if is_master():
        logger.info(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        logger.info(f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            logger.info(f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}")
        
        logger.info(f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            logger.info(f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}")
    # ------------------------------
    # Define optimizer groups
    # ------------------------------
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
    # ------------------------------
    # device
    # ------------------------------
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()
    logger.info(f"Using device: {device}")
    # ------------------------------
    # model prepare
    # ------------------------------
    # model to device
    model.to(device)

    # model compile
    if train_cfg.compile:
        model = torch.compile(model)
    
    # distributed training
    if is_dist():
        model = wrap_model(model)
    # ------------------------------
    # model training
    # ------------------------------
    epoch = 0
    epoch_times = []
    best_val_loss = float("inf")
    global_step = 0
    # training stats accumulators collector
    accumulated_stats = {
        "token_per_second": [],
        "data_load_time": [],
        "fw_bw_time": [],
        "post_process_time": [],
        "images_per_sample": [],
    }
    while global_step < train_cfg.max_training_steps:
        #! epoch start
        epoch_start_time = time.time()
        epoch += 1
        # loss and token collector
        total_train_loss = 0
        total_tokens_processed = 0
        # init model and optimizer state
        model.train()
        optimizer.zero_grad()

        #! data load start
        data_load_start = time.time()
        for i, batch in enumerate(synchronized_dataloader_step(train_loader, is_dist())):
            is_update_step = ((i + 1) % train_cfg.gradient_accumulation_steps == 0) or ((i + 1) == len(train_loader)) 
            #! batch start
            batch_start_time = time.time()

            # ------------------------------
            # data load
            # ------------------------------
            # input data
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            #! data load end
            data_load_time = time.time() - data_load_start 
            # ------------------------------
            # Forward and Backward
            # ------------------------------
            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (is_dist() and train_cfg.gradient_accumulation_steps > 1 and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()
            
            #! forward and backward start
            fw_bw_start = time.time()

            # forward
            autocast_context = torch.autocast(
                device_type=device.type, 
                dtype=torch.bfloat16 if device.type in ["cuda", "cpu"] else torch.float16
            )
            with autocast_context:
                with context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            
            # backward
            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps
            loss.backward()

            #! forward and backward end
            fw_bw_time = time.time() - fw_bw_start
            # ------------------------------
            # Post process
            # ------------------------------
            #! post process start
            post_process_start = time.time()

            # model weights update
            if is_update_step:
                # gradient clipping
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)
                
                # adjust mp learning rate
                adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                optimizer.param_groups[0]["lr"] = adj_lr_mp
                
                # adjust backbones learning rate
                if train_cfg.lr_backbones > 0:
                    adj_lr_backbones = get_lr(global_step, train_cfg.lr_backbones, train_cfg.max_training_steps)
                    optimizer.param_groups[1]["lr"] = adj_lr_backbones
                
                # update weights
                optimizer.step()
                # zero gradients
                optimizer.zero_grad()
            
            # loss collection
            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            # token processed collection
            num_tokens = torch.sum(attention_mask).item()  # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens

            #! post process end
            post_process_time = time.time() - post_process_start 
            #! batch end
            batch_end_time = time.time()

            # Multiply by world size to get global tokens/s
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size() * num_tokens / batch_duration

            # images per sample
            images_per_sample = [len(image_pack) for image_pack in images]

            # Accumulate training stats
            accumulated_stats["tokens_per_second"].append(tokens_per_second)
            accumulated_stats["data_load_time"].append(data_load_time)
            accumulated_stats["fw_bw_time"].append(fw_bw_time)
            accumulated_stats["post_process_time"].append(post_process_time)
            accumulated_stats["images_per_sample"].extend(images_per_sample)
            # ------------------------------
            # Model Valid
            # ------------------------------
            if train_cfg.eval_in_epochs and (global_step % train_cfg.eval_interval == 0) and is_update_step and (global_step > 0):
                # model valid mode
                model.eval()
                
                # model cuda memory clear
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # model valid and eval
                with torch.no_grad():
                    # ------------------------------
                    # Valid
                    # ------------------------------
                    total_val_loss = 0
                    for batch in val_loader:
                        # input data
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        # forward
                        with autocast_context:
                            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
                        # loss collection
                        total_val_loss += loss.item()
                    # valid loss
                    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                    avg_val_loss = statistics.mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    # model save
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            save_model = model.module if is_dist() else model  # unwrap the model for saving if DDP
                            save_model.save_pretrained(save_directory=Path(vlm_cfg.vlm_checkpoint_path).joinpath(run_name))
                    # ------------------------------
                    # Evaluate
                    # ------------------------------
                    lmms_results = {}
                    if train_cfg.use_lmms_eval:
                        from evaluation import cli_evaluate
                        # Evaluate args
                        eval_args = argparse.Namespace(
                            model=model.module if is_dist() else model,
                            tasks=train_cfg.lmms_eval_tasks,
                            limit=train_cfg.lmms_eval_limit,
                            batch_size=train_cfg.lmms_eval_batch_size,
                            process_with_media=True,
                            device=device,
                        )
                        # Evaluate using the CLI wrapper
                        eval_results = cli_evaluate(eval_args)
                        # Evaluate results collection
                        if is_master() and eval_results and ("results" in eval_results[0]):
                            for task_name, task_results in eval_results[0]["results"].items():
                                for metric_name, metric_value in task_results.items():
                                    if isinstance(metric_value, (int, float)):
                                        lmms_results[f"{task_name}_{metric_name.split(",")[0]}"] = metric_value
                    # ------------------------------
                    # Valid and Evaluate Log
                    # ------------------------------
                    if is_master():
                        logger.info(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")
                        if train_cfg.log_wandb:
                            run.log({
                                "val_loss": avg_val_loss,
                                **{f"lmms_eval/{key}": value for key, value in lmms_results.items()}
                            }, step=global_step)
                
                # model evaluate mode
                model.train()
            # ------------------------------
            # Log training stats every N steps (ALl RANKS must participate in collective ops)
            # ------------------------------
            if global_step % train_cfg.stats_log_interval == 0 and len(accumulated_stats["tokens_per_second"]) > 0 and is_update_step:
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in ["tokens_per_second", "data_load_time", "fw_bw_time", "post_process_time", "images_per_sample"]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]  # Flatten list of lists
                        stats[f"avg_{key}"] = statistics.mean(all_values_flat)
                    else:
                        stats[f"avg_{key}"] = statistics.mean(accumulated_stats[key])
                
                for key in ["data_load_time", "fw_bw_time", "post_process_time", "images_per_sample"]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]
                        stats[f"max_{key}"] = max(all_values_flat)
                    else:
                        stats[f"max_{key}"] = max(accumulated_stats[key])
                
                if is_dist():
                    all_images_values = dist_gather(accumulated_stats["images_per_sample"])
                    all_images_flat = [item for sublist in all_images_values for item in sublist]
                    stats["min_images_per_sample"] = min(all_images_flat)
                else:
                    stats["min_images_per_sample"] = min(accumulated_stats["images_per_sample"])
                
                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        **{f"training_stats/{key}": value for key, value in stats.items()},
                    }, step=global_step)
                # All RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []
            # ------------------------------
            # Batch Log
            # ------------------------------
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_dist():
                    batch_loss_gathered = statistics.mean(dist_gather(batch_loss))
                else:
                    batch_loss_gathered = batch_loss
                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        "batch_loss": batch_loss_gathered,
                        **({"grad_norm": grad_norm} if train_cfg.max_grad_norm is not None else {}),
                    }, step=global_step)
            # ------------------------------
            # Training stop
            # ------------------------------
            if is_update_step:
                global_step += 1
                if global_step >= train_cfg.max_training_steps:
                    break
            
            #! data load start
            data_load_start = time.time()

        # Gather average batch loss from all ranks if DDP
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_loss = statistics.mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss

        #! epoch end
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # Gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        # Epoch Log
        if is_master():
            if train_cfg.log_wandb:
                run.log({
                    "epoch_loss": avg_train_loss,
                    "epoch_duration": epoch_duration,
                    "epoch_tokens_per_second": epoch_tokens_per_second,
                })
            logger.info(f"Epoch: {epoch}, Step: {global_step}/{train_cfg.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f} | T/s: {epoch_tokens_per_second:.2f}")
    # ------------------------------
    # Summary Statistics
    # ------------------------------
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        logger.info(f"Average time per epoch: {avg_epoch_time:.2f}s")
        
        total_training_time = sum(epoch_times)
        batch_size = int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        logger.info(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in config!)
        if vlm_cfg.hf_repo_name is not None:
            logger.info(f"Training complete. Pushing model to Hugging Face Hub...")
            hf_model = VisionLanguageModel.from_pretrained(Path(vlm_cfg.vlm_checkpoint_path).joinpath(run_name))
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)
        
        # wandb log
        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()




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
