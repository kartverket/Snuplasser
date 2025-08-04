import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


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
        img = np.array(Image.open(os.path.join(self.image_dir, f"image_{file_id}.png")).convert("RGB"))

        if self.dom_dir is not None:
            dom = np.array(Image.open(os.path.join(self.dom_dir, f"dom_{file_id}.png")).convert("L"))
            img = np.concatenate([img, dom[..., None]], axis=-1)

        mask = None
        if self.mask_dir is not None:
            m = np.array(Image.open(os.path.join(self.mask_dir, f"mask_{file_id}.png")).convert("L")) // 255
            mask = m

        if self.transform:
            data = {"image": img}
            if mask is not None:
                data["mask"] = mask
            out = self.transform(**data)
            img = out["image"]
            mask = out.get("mask", mask)

        if isinstance(img, torch.Tensor):
            image_tensor = img.float()
        else:
            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        mask_tensor = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                t = mask.float()
            else:
                t = torch.from_numpy(mask).float()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            mask_tensor = t

        return image_tensor, mask_tensor, f"{file_id}.png"