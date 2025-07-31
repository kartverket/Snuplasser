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
        image_path = os.path.join(self.image_dir, f"{file_id}.png")
        mask_path = os.path.join(self.mask_dir, f"mask_{file_id[6:]}.png")
        dom_path = os.path.join(self.dom_dir, f"dom_{file_id[6:]}.png")

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
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        filename = f"{file_id}.png"
        return image, mask, filename 


#hjelpe funksjon hvis det trenges
def load_ignore_ids(missing_file_path):
    ignore_ids = set()
    with open(missing_file_path, "r") as f:
        for line in f:
            filename = line.strip()
            if filename.startswith("image_"):
                file_id = Path(filename).stem
            elif filename.startswith("mask_"):
                file_id = "image_" + Path(filename).stem.replace("mask_", "")
            elif filename.startswith("dom_"):
                file_id = "image_" + Path(filename).stem.replace("dom_", "")
            else:
                continue
            ignore_ids.add(file_id)
    return ignore_ids


class SnuplassPredictDataset(Dataset):
    def __init__(self, image_dir, dom_dir, transform=None):
        self.image_dir = image_dir
        self.dom_dir = dom_dir
        self.transform = transform

        files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.startswith("image_") and f.endswith(".png")
            ]
        )
        self.file_list = [Path(f).stem for f in files]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        img_path = os.path.join(self.image_dir, f"{file_id}.png")
        dom_path = os.path.join(self.dom_dir, f"dom_{file_id[6:]}.png")

        img = np.array(Image.open(img_path).convert("RGB"))
        dom = np.array(Image.open(dom_path).convert("L"))
        dom = np.expand_dims(dom, axis=-1)  # (H,W,1)
        image = np.concatenate((img, dom), axis=-1)  # (H,W,4)

        if self.transform:
            augmented = self.transform(image=np.array(image))
            image = augmented["image"]

        filename = f"{file_id}.png"
        return image, filename


def load_numpy_split_stack(
    image_dir, mask_dir, dom_dir, holdout_size=5, test_size=0.2, seed=42,  ignore_file_path=None 
):
    """
    Laster inn hele datasettet som numpy-arrays, splitter i tren/val/test og returnerer stacks.
    """
    np.random.seed(seed)

    all_files = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.startswith("image_") and f.endswith(".png")
        ]
    )
    file_ids = [Path(f).stem for f in all_files]
    
    if ignore_file_path is not None:
        ignore_ids = load_ignore_ids(ignore_file_path)
        file_ids = [fid for fid in file_ids if fid not in ignore_ids]

    if len(file_ids) < holdout_size + 2:
        raise ValueError(
            "For få bilder til å gjennomføre splitting med holdout og validering."
        )

    np.random.shuffle(file_ids)
    holdout_ids = file_ids[:holdout_size]
    remaining_ids = file_ids[holdout_size:]

    train_ids, val_ids = train_test_split(
        remaining_ids, test_size=test_size, random_state=seed
    )
   

    return train_ids, val_ids, holdout_ids