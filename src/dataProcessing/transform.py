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
            A.RandomRotate90(p=0.5), #  Sannsynlighet for å rotere bildet 90 grader
            A.RandomBrightnessContrast(
                brightness_limit=cfg["brightness_limit"],
                contrast_limit=cfg["contrast_limit"],
                p=0.5,
            ),  # Endringer i solforhold, årstid, skygge eller skydetthet
            ToTensorV2(),
        ]
    )
    return base_transform


def get_val_transforms():
    return A.Compose([ToTensorV2()])