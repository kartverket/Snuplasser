import albumentations as A

class Augmentation:
    def __init__(self, channels):
        self.augmentation = self.get_augmentation(channels)
    
    def __call__(self, image, mask):
        augmented = self.augmentation(image=image, mask=mask)
        return augmented["image"], augmented["mask"]
    
    def get_augmentation(self, channels):
        transforms = [
            # Spatial transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        ]

        if channels == 3:
            transforms.append(
                A.OneOf([
                    A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=2,
                        shadow_dimension=4,
                        p=0.8
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 1),
                        angle_lower=0,
                        angle_upper=1,
                        num_flare_circles_lower=6,
                        num_flare_circles_upper=10,
                        p=0.5
                    ),
                ], p=1.0)),
            transforms.append(
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.8),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
                ], p=0.5))
            transforms.append(
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ], p=0.3))

        transforms.append(
            A.Normalize(mean=[0.485] * channels, std=[0.229] * channels, max_pixel_value=255.0)
        )

        return A.Compose(
            transforms,
            p=0.8,
            additional_targets={"mask": "mask"}
        )