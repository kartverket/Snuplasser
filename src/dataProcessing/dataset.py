import os
import random
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import json
from datetime import datetime


class SnuplassDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dom_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dom_dir = dom_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        
        image_path = os.path.join(self.image_dir, f"image_{file_id}.png")
        dom_path = os.path.join(self.dom_dir, f"dom_{file_id}.png")
        mask_path = os.path.join(self.mask_dir, f"mask_{file_id}.png")


        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        dom = Image.open(dom_path).convert("L")

        dom = np.expand_dims(dom, axis=-1)  # (H, W, 1)
        image = np.concatenate((image, dom), axis=-1)  # (H, W, 4)

        if self.transform:
            augmented = self.transform(
                image=np.array(image),
                mask=np.array(mask) // 255,
            )
            image = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).float()
        if mask.max() > 1:
            mask = mask / 255.0
        if mask.ndim==2:
            mask = mask.unsqueeze(0)

        filename = f"{file_id}.png"
        return image, mask, filename 