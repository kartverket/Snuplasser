import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class SnuplassDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dom_dir, split="train", transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dom_dir = dom_dir
        self.split = split
        self.transform = transform

        if self.split == "train":
            self.file_list = load_numpy_split_stack(image_dir=image_dir)[0]
        else:
            self.file_list = load_numpy_split_stack(image_dir=image_dir)[1]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        image_path = os.path.join(self.image_dir, f"{file_id}.png")
        mask_path = os.path.join(self.mask_dir, f"mask_{file_id[6:]}.png")
        dom_path = os.path.join(self.dom_dir, f"dom_{file_id[6:]}.png")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        dom = np.array(Image.open(dom_path).convert("L"))

        dom = np.expand_dims(dom, axis=-1)  # (H, W, 1)
        image = np.concatenate((image, dom), axis=-1)  # (H, W, 4)

        if self.transform:
            image, mask = self.transform(image=image, mask=mask)

        return image, mask
    
    def __getfilename__(self, idx):
        return self.file_list[idx]


def load_numpy_split_stack(
    image_dir, holdout_size=5, test_size=0.2, seed=42
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