import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
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

    @staticmethod
    def create_train_val_split(
        image_dir="data/images",
        train_file="data/splits/train.txt",
        val_file="data/splits/val.txt",
        split_ratio=0.8,
        seed=42,
    ):
        import random

        random.seed(seed)

        image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.startswith("image_") and f.endswith(".png")
            ]
        )
        file_ids = [Path(f).stem for f in image_files]

        random.shuffle(file_ids)
        split_index = int(len(file_ids) * split_ratio)
        train_ids = file_ids[:split_index]
        val_ids = file_ids[split_index:]

        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, "w") as f:
            f.writelines([id_ + "\n" for id_ in train_ids])
        with open(val_file, "w") as f:
            f.writelines([id_ + "\n" for id_ in val_ids])

        print(f"✅ Laget split: {len(train_ids)} train, {len(val_ids)} val")

        meta = {
            "created": datetime.now().isoformat(),
            "seed": seed,
            "split_ratio": split_ratio,
            "total_files": len(file_ids),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "source_dir": image_dir,
        }
        meta_path = Path(train_file).parent / "split_meta.json"
        with open(meta_path, "w") as meta_f:
            json.dump(meta, meta_f, indent=2)

        print(f"\n✅ Laget split: {len(train_ids)} train, {len(val_ids)} val")
        print(f"📄 Metadata lagret i: {meta_path}")


if __name__ == "__main__":
    # Eksempel på hvordan du kan bruke dataset-klassen
    dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        file_list="data/splits/train.txt",
        transform=None,  # Her kan du legge til eventuelle transformasjoner
    )

    print(f"Totalt antall bilder i datasettet: {len(dataset)}")

    # Eksempel på å hente et bilde og maske
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

    # Lag en trenings- og valideringssplit
    SnuplassDataset.create_train_val_split()
