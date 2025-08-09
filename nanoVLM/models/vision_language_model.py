# -*- coding: utf-8 -*-

# ***************************************************
# * File        : vision_language_model.py
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
import json
import tempfile
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from nanoVLM.data_provider.processors import get_tokenizer
from nanoVLM.models.utils import top_k_top_p_filtering
from nanoVLM.models.config import VLMConfig
from nanoVLM.models.vision_transformer import ViT
from nanoVLM.models.language_model import LanguageModel
from nanoVLM.models.modality_projector import ModalityProjector

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class VisionLanguageModel(nn.Module):

    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()

        self.cfg = cfg
        
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    
    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()
        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = (input_ids == self.tokenizer.image_token_id)
        updated_token_embd[mask] = image_embd \
            .view(-1, image_embd.size(-1)) \
            .to(updated_token_embd.dtype)
        
        return updated_token_embd
    
    def forward(self, input_ids, images, attention_mask=None, targets=None):
        pass
    
    @torch.inference_mode()
    def generate(self, 
                 input_ids, 
                 images, 
                 attention_mask=None, 
                 max_new_tokens=5, 
                 top_k=50, 
                 top_p=0.9, 
                 temperature=0.5, 
                 greedy=False):
        if isinstance(images, list):
            if not images:
                images = torch.empty(
                    0, 
                    self.cfg.vit_channels, 
                    self.cfg.vit_image_size, 
                    self.cfg.vit_image_size, 
                    device=input_ids.device
                )
    
    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, *, revision: Optional[str]=None) -> "VisionLanguageModel":
        pass

    def save_pretrained(self, save_directory: str) -> None:
        pass

    def push_to_hub(self, repo_id: str, private: bool=False) -> None:
        pass


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
