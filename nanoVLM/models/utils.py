# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
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
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def seed_torch(seed: int=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def destory_dist():
    dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return dist.get_rank() == 0 if is_dist() else True


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)

    return o_all


def wrap_model(model):
    return DDP(model, device_ids=[dist.get_rank()])


def check_multiple_choice_with_regex(model_outputs, correct_answers):
    """
    Used to check our models performance on multiple choice tasks. 
    This can also be done in a more involved way with e.g. LLM-as-a-judge

    Args:
        model_outputs (_type_): _description_
        correct_answers (_type_): _description_
    """
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        # Strip any trailing nwelines and convert to uppercase
        correct_answer = correct_answer.rstrip("\n").upper()
        # Look for the answer letter at the begining of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # answer within parentheses
        ]
        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.

    Args:
        logits (_type_): _description_
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 1.0.
        filter_value (_type_, optional): _description_. Defaults to -float("Inf").
    """
    top_k = min(top_k, logits.size(-1))  # Safety
    if top_k > 0:
        # remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # always keep the first token
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits




# 测试代码 main 函数
def main():
    print(torch.initial_seed())
    print(torch.initial_seed() % 2**32)

if __name__ == "__main__":
    main()
