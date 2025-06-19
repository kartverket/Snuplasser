import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transform import get_train_transforms
from config import augmentation_profiles


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


def visualize_multiple_augmentations(image_path, mask_path, cfg_name="basic", n=4):
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path)) // 255
    transform = get_train_transforms(augmentation_profiles[cfg_name])

    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        augmented = transform(image=image, mask=mask)
        image_aug = augmented["image"].permute(1, 2, 0).numpy()
        mask_aug = augmented["mask"].numpy()

        axes[i][0].imshow(image_aug)
        axes[i][0].set_title(f"Augmentert bilde {i+1}")
        axes[i][1].imshow(mask_aug, cmap="gray")
        axes[i][1].set_title(f"Maske {i+1}")

        for a in axes[i]:
            a.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_multiple_augmentations(
        image_path="data\images\image_249322_6786313_249385_6786382.png",
        mask_path="data\masks\mask_249322_6786313_249385_6786382.png",
        cfg_name="basic",  # Se config
        n=4
    )
