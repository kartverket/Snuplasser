import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class EndepunktDataset(Dataset):
    def __init__(self, image_dir, dom_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.dom_dir = Path(dom_dir)
        self.image_files = sorted([f for f in self.image_dir.glob("endepunkt_*.png")])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        file_id = image_path.stem  
        dom_filename = file_id.replace("endepunkt_", "endepunkt_dom") + ".png"
        dom_path = self.dom_dir / dom_filename

       
        image = Image.open(image_path).convert("RGB")
        dom = Image.open(dom_path).convert("L")

     
        image_np = np.array(image)
        dom_np = np.expand_dims(np.array(dom), axis=-1)

        combined = np.concatenate([image_np, dom_np], axis=-1)

        if self.transform:
            augmented = self.transform(image=combined)
            combined = augmented["image"]

        if not isinstance(combined, torch.Tensor):
            combined = torch.from_numpy(combined).permute(2, 0, 1).float()

        return combined
    

def get_dataset(image_dir, dom_dir, transform=None):
    return EndepunktDataset(image_dir, dom_dir, transform=transform)
