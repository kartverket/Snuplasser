import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import augmentation_profiles
import matplotlib.pyplot as plt


class SnuplassDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        with open(file_list, 'r') as f:
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


def get_train_transforms(cfg):
    return A.Compose([
        A.HorizontalFlip(p=cfg["flip_p"]),
        A.RandomRotate90(p=cfg["rot90_p"]),
        A.RandomBrightnessContrast(p=cfg["brightness_p"]),  # Endringer i solforhold, årstid, skygge eller skydetthet
        A.ShiftScaleRotate(  
            shift_limit=cfg["shift"],
            scale_limit=cfg["scale"],
            rotate_limit=cfg["rotate"],
            p=cfg["ssr_p"]
        ),  # 	Simulerer variasjoner i flyhøyde, bildestabilisering
        A.Normalize(),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])


def visualize_augmented_sample(image_path, mask_path, cfg_name="basic"):
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path)) // 255

    transform = get_train_transforms(augmentation_profiles[cfg_name])
    augmented = transform(image=image, mask=mask)

    image_aug = augmented["image"].permute(1, 2, 0).numpy()
    mask_aug = augmented["mask"].numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_aug)
    ax[0].set_title("Transformert bilde")
    ax[1].imshow(mask_aug, cmap="gray")
    ax[1].set_title("Transformert maske")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()