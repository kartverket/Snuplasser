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
            raise ValueError(
                f"Invalid entry in file_list. Expected 2 or 3 elements, got {len(entry)}"
            )

        # Load image and DOM
        img = np.array(Image.open(image_path).convert("RGB"))
        dom = np.array(Image.open(dom_path).convert("L"))
        # Append DOM as extra channel
        img = np.concatenate([img, dom[..., None]], axis=-1)

        # Load mask or create dummy
        if mask_path:
            mask = np.array(Image.open(mask_path).convert("L")) // 255
        else:
            # dummy mask of zeros at same HxW
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Apply transforms if any
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Convert image to tensor (C, H, W), float
        if isinstance(img, torch.Tensor):
            image_tensor = img.float()
        else:
            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Convert mask to tensor (1, H, W), float
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()
        else:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # Filename for identification
        filename = Path(image_path).name

        return image_tensor, mask_tensor, filename
