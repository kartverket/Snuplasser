import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg: dict = None, ratio: float = None, height=512, width=512):
    return A.Compose(
        [
            A.RandomResizedCrop(
                height, width, scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.8
            ),
            A.Rotate(limit=45, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            ToTensorV2(),
        ]
    )
