import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os


class SnuplassDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        with open(file_list, "r") as f:
            self.files = [line.strip() for line in f.readlines()]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]
        mask_file_id = file_id.replace("image", "mask")
        image_path = os.path.join(self.image_dir, f"{file_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{mask_file_id}.png")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path)) // 255  # binær 0/1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, torch.from_numpy(mask).long()
