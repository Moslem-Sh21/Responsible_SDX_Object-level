# This project uses Stable Diffusion XL, a model developed by Stability AI

import logging
import math
import os
import random
import torch.nn as nn
from pathlib import Path
from typing import Iterable, Optional
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from model import model_types
from config import parse_args
from utils_model import save_model, load_model, load_weights
from utils_data import get_dataloader

logger = get_logger(__name__)


def weighted_concept_loss(predicted_noise, true_noise, daam_heatmaps, alpha=1.0, beta=0.1):
    """
    A loss function that applies higher weights to regions identified in DAAM heatmaps.

    Args:
        predicted_noise: The noise predicted by the model (with concept vector c)
        true_noise: The target noise
        daam_heatmaps: Attention maps highlighting regions related to the concept
        alpha: Base weight for all pixels
        beta: Additional weight multiplier for concept-related regions

    Returns:
        Weighted L2 loss
    """
    if daam_heatmaps.max() > 1.0:
        daam_heatmaps = daam_heatmaps / 255.0

    batch_min = daam_heatmaps.view(daam_heatmaps.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    batch_max = daam_heatmaps.view(daam_heatmaps.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    daam_heatmaps = (daam_heatmaps - batch_min) / (batch_max - batch_min + 1e-8)
    weights = alpha + (beta * daam_heatmaps)
    squared_error = (predicted_noise - true_noise) ** 2
    weighted_error = weights * squared_error

    return weighted_error.mean()


def unfreeze_layers_unet(unet):
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s