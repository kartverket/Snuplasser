import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg):
    return A.Compose([
        A.HorizontalFlip(p=cfg["flip_p"]),
        A.RandomRotate90(p=cfg["rot90_p"]),
        A.RandomBrightnessContrast(p=cfg["brightness_p"]),  # Endringer i solforhold, årstid, skygge eller skydetthet
        # A.ShiftScaleRotate(  
        #     shift_limit=cfg["shift"],
        #     scale_limit=cfg["scale"],
        #     rotate_limit=cfg["rotate"],
        #     p=cfg["ssr_p"]
        # ),  # 	Simulerer variasjoner i flyhøyde, bildestabilisering
        A.Normalize(),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])