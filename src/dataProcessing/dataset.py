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
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if self.dom_dir is not None:
            dom_path = os.path.join(self.dom_dir, f"dom_{file_id}.png")
            dom = Image.open(dom_path).convert("L")
            dom_np = np.expand_dims(np.array(dom), axis=-1)
            image_np = np.concatenate((image_np, dom_np), axis=-1)

        mask_np = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, f"mask_{file_id}.png")
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask) // 255

        if self.transform:
            data = {"image": image_np}
            if mask_np is not None:
                data["mask"] = mask_np
            augmented = self.transform(**data)
            image_np = augmented["image"]
            mask_np = augmented.get("mask", mask_np)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        if mask_np is not None:
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            mask_tensor = None

        filename = f"{file_id}.png"
        return image_tensor, mask_tensor, filename


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
    image_dir, mask_dir, dom_dir, holdout_size=5, test_size=0.2, seed=42
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