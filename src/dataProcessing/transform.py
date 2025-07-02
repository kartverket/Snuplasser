import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg: dict, ratio: float | None = None):
    if ratio is None:
        return A.Compose([ToTensorV2()])

    if ratio < 0 or ratio > 1:
        raise ValueError(f"Ratio must be between 0 and 1. Received: {ratio}")

    base_transform = A.Compose(
        [
            A.HorizontalFlip(p=cfg["flip_p"]),
            A.RandomRotate90(degrees=90),
            A.RandomBrightnessContrast(
                brightness=cfg["brightness_p"]
            ),  # Endringer i solforhold, årstid, skygge eller skydetthet
            ToTensorV2(),
        ]
    )
    return base_transform


def get_val_transforms():
    return A.Compose([ToTensorV2()])