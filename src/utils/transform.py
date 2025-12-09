import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg: dict = None, ratio: float = None) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomCrop(height=512, width=512, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
            A.CLAHE(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
            A.GridDistortion(p=0.2),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            ToTensorV2(),
        ]
    )
