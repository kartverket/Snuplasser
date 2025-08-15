from pathlib import Path
from PIL import Image
import numpy as np
import torch


def compute_loss_weights(mask_dir: str) -> tuple[float, float, torch.Tensor]:
    """
    Beregner vekter for Dice + BCE loss basert på fordelingen av svarte og hvite piksler i maskene.
    Argumenter:
        mask_dir (str): Katalog som inneholder maskebilder (binære PNG-filer).
    Returnerer:
        tuple: (dice_weight, bce_weight, pos_weight)
    """
    black_pixels = 0
    white_pixels = 0

    for path in Path(mask_dir).glob("*.png"):
        arr = np.array(Image.open(path).convert("L"))
        white_pixels += np.sum(arr == 255)
        black_pixels += np.sum(arr == 0)

    total_pixels = white_pixels + black_pixels

    # Vekter til Dice + BCE
    dice_weight = black_pixels / total_pixels
    bce_weight = white_pixels / total_pixels

    # pos_weight til BCEWithLogitsLoss
    pos_weight = torch.tensor(black_pixels / white_pixels)

    return dice_weight, bce_weight, pos_weight