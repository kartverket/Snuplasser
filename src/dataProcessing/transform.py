# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# """
# Transformasjoner for trenings- og valideringsdata.
# Valg:
# - Ingen normalisering benyttes. Dette er bevisst for å unngå dataleakage og sikre at modellen fungerer på nye bilder uten behov for global statistikk.
# - Augmentering gjøres kun under trening for å gjøre modellen robust mot variasjoner i solforhold, skyggemønster og bildeorientering.
# - `ToTensorV2()` konverterer NumPy-arrays til PyTorch-tensorer, som forventet av modellen.
# """


# def get_train_transforms(cfg: dict, ratio: float | None = None):
#     if ratio is None:
#         return A.Compose([ToTensorV2()])

#     if ratio < 0 or ratio > 1:
#         raise ValueError(f"Ratio must be between 0 and 1. Received: {ratio}")

#     base_transform = A.Compose(
#         [
#             A.HorizontalFlip(p=cfg["flip_p"]),
#             A.RandomRotate90(p=cfg["rot90_p"]),
#             A.RandomBrightnessContrast(
#                 p=cfg["brightness_p"]
#             ),  # Endringer i solforhold, årstid, skygge eller skydetthet
#             ToTensorV2(),
#         ]
#     )

#     return A.OneOf(
#         [
#             base_transform,
#             A.NoOp(),
#         ],
#         p=ratio,
#     )


# def get_val_transforms():
#     return A.Compose([ToTensorV2()])
