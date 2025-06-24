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
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if not os.path.exists(file_list):
            raise FileNotFoundError(f"Filen {file_list} finnes ikke.")

        if os.path.getsize(file_list) == 0:
            print(f"⚠️ Advarsel: {file_list} er tom – datasettet vil være tomt.")

        with open(file_list, "r") as f:
            self.file_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_id = self.file_names[idx]
        image_path = os.path.join(self.image_dir, f"{file_id}.png")
        mask_path = os.path.join(self.mask_dir, f"mask_{file_id[6:]}.png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            augmented = self.transform(
                image=np.array(image), mask=np.array(mask) // 255
            )
            image = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask) / 255).unsqueeze(0).float()

        return image, mask

    @staticmethod
    def create_train_val_split(
        image_dir="data/images",
        train_file="data/splits/train.txt",
        val_file="data/splits/val.txt",
        split_ratio=0.8,
        seed=42,
    ):
        random.seed(seed)

        image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.startswith("image_") and f.endswith(".png") and f.strip()
            ]
        )
        file_ids = [Path(f).stem for f in image_files if f.strip()]

        if not file_ids:
            raise ValueError("Ingen bildefiler funnet i angitt mappe.")

        random.shuffle(file_ids)
        split_index = int(len(file_ids) * split_ratio)
        train_ids = file_ids[:split_index]
        val_ids = file_ids[split_index:]

        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, "w") as f:
            f.writelines([id_ + "\n" for id_ in train_ids])
        with open(val_file, "w") as f:
            f.writelines([id_ + "\n" for id_ in val_ids])

        metadata = {
            "created": datetime.now().isoformat(),
            "seed": seed,
            "split_ratio": split_ratio,
            "total_files": len(file_ids),
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "train_file": train_file,
            "val_file": val_file,
        }

        meta_path = os.path.join(os.path.dirname(train_file), "split_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Laget split: {len(train_ids)} train, {len(val_ids)} val")
        print(f"ℹ️ Metadata lagret i {meta_path}")


def load_numpy_split_stack(image_dir, mask_dir, holdout_size=5, test_size=0.2, seed=42):
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

    def load_stack(ids):
        images, masks = [], []
        for file_id in ids:
            img_path = os.path.join(image_dir, f"{file_id}.png")
            mask_path = os.path.join(mask_dir, f"mask_{file_id[6:]}.png")

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"⛔️ Fil mangler: {img_path} eller {mask_path} – hoppes over")
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path)) // 255

            images.append(image)
            masks.append(mask)

        if not images:
            raise ValueError("Ingen gyldige bilde/mask-par funnet for subset.")

        return np.stack(images), np.stack(masks)

    X_train, y_train = load_stack(train_ids)
    X_val, y_val = load_stack(val_ids)
    X_test, y_test = load_stack(holdout_ids)

    print(
        f"📦 Treningsdata: {len(X_train)} | Validering: {len(X_val)} | Test (holdout): {len(X_test)}"
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    SnuplassDataset.create_train_val_split()
