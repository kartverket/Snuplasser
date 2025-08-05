import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class SnuplassDataset(Dataset):
    def __init__(self, file_list: list[tuple], transform=None):
        """
        file_list: List of tuples with paths.
            - Training: List of (image_path, dom_path, mask_path)
            - Predict:  List of (image_path, dom_path)
        transform: Albumentations or similar with signature transform(image, mask)
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        entry = self.file_list[idx]
        # Unpack tuple
        if len(entry) == 3:
            image_path, dom_path, mask_path = entry
        elif len(entry) == 2:
            image_path, dom_path = entry
            mask_path = None
        else:
            raise ValueError(f"Invalid entry in file_list. Expected 2 or 3 elements, got {len(entry)}")

        # Load image and dom
        img = np.array(Image.open(image_path).convert("RGB"))
        dom = np.array(Image.open(dom_path).convert("L"))
        # Append dom as extra channel
        img = np.concatenate([img, dom[..., None]], axis=-1)

        # Load mask if present
        mask = None
        if mask_path is not None:
            m = np.array(Image.open(mask_path).convert("L")) // 255
            mask = m

        # Apply transform
        if self.transform:
            data = {"image": img}
            if mask is not None:
                data["mask"] = mask
            out = self.transform(**data)
            img = out["image"]
            mask = out.get("mask", mask)

        # Convert to tensors
        # Image: CHW float
        if isinstance(img, torch.Tensor):
            image_tensor = img.float()
        else:
            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Mask: add channel dim
        mask_tensor = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                t = mask.float()
            else:
                t = torch.from_numpy(mask).float()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            mask_tensor = t

        # Return image, mask (or None), and source path for identification
        return image_tensor, mask_tensor, Path(image_path).name
