import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg: dict, ratio: float | None = None):

    if ratio is None:
        return A.Compose(
        [
            ToTensorV2(),
        ]
    )

    if ratio < 0 or ratio > 1:
        raise ValueError(f"Ratio must be between 0 and 1. Received: {ratio}")

    base_transform = A.Compose(
        [
            A.HorizontalFlip(p=cfg.get("flip_p", 0.5)),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),  #  Sannsynlighet for å rotere bildet 90 grader
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=cfg.get("scale_limit", 0.1),
                rotate_limit=cfg.get("rotate_limit", 15),
                p=0.5,
            ),  # Endringer i posisjon
            A.RandomBrightnessContrast(
                brightness_limit=cfg.get("brightness_limit", 0.2),
                contrast_limit=cfg.get("contrast_limit", 0.2),
                p=0.5,
            ),  # Endringer i solforhold, årstid, skygge eller skydetthet
            A.GaussianBlur(blur_limit=(3, 5), p=cfg.get("gaussian_blur_p", 0.2)),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=30,
                p=cfg.get("elastic_transform_p", 0.3),
            ),
            ToTensorV2(),
        ]
    )
    return base_transform


def get_val_transforms(cfg: dict):
    return A.Compose(
        [
            ToTensorV2(),
        ]
    )