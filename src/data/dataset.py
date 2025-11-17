from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class SnuplassDataset(Dataset):
    """
    Laster inn dataen til et datasett.
    Argumenter:
        file_list (list): liste med tupler med filer.
            - Training: liste med (image_path, dom_path, mask_path)
            - Predict:  liste med (image_path, dom_path)
        transform: transformasjoner til å brukes på data
    """
    def __init__(self, file_list: list[tuple], transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        entry = self.file_list[idx]
        if len(entry) == 3:
            image_path, dom_path, mask_path = entry
        elif len(entry) == 2:
            image_path, dom_path = entry
            mask_path = None
        else:
            raise ValueError(
                f"Ulovlig antall elementer i entry, forventet 2 eller 3, men fant {len(entry)}"
            )

        # Last inn bilde og DOM
        img = np.array(Image.open(image_path).convert("RGB"))
        dom = np.array(Image.open(dom_path).convert("L"))
        # Legg til DOM som en ekstra kanal på bildet
        img = np.concatenate([img, dom[..., None]], axis=-1)

        # Last inn maske, eller generer en svart maske dersom det ikke finnes en
        if mask_path:
            mask = np.array(Image.open(mask_path).convert("L")) // 255
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Transformer hvis det er gitt en transformasjon
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Lag tensor og konverter til flyttall
        if isinstance(img, torch.Tensor):
            image_tensor = img.float()
        else:
            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()
        else:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        filename = Path(image_path).name

        return image_tensor, mask_tensor, filename
    
class HelicopterDataset(Dataset):
    """
    Laster inn dataen til et datasett.
    Argumenter:
        file_list (list): liste med tupler med filer.
            - Training: liste med (image_path, dom_path, mask_path)
            - Predict:  liste med (image_path, dom_path)
        transform: transformasjoner til å brukes på data
    """
    def __init__(self, file_list: list[tuple], transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        entry = self.file_list[idx]
        if isinstance(entry, tuple) and len(entry) == 2:
            image_path, mask_path = entry
        elif isinstance(entry, str):
            image_path = entry
            mask_path = None
        else:
            raise ValueError(
                f"Ulovlig antall elementer i entry, forventet 2 eller 3, men fant {len(entry)}"
            )

        # Last inn bilde og DOM
        img = np.array(Image.open(image_path).convert("RGB"))

        # Last inn maske, eller generer en svart maske dersom det ikke finnes en
        if mask_path:
            mask = np.array(Image.open(mask_path).convert("L")) // 255
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Transformer hvis det er gitt en transformasjon
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Lag tensor og konverter til flyttall
        if isinstance(img, torch.Tensor):
            image_tensor = img.float()
        else:
            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()
        else:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        filename = Path(image_path).name

        return image_tensor, mask_tensor, filename