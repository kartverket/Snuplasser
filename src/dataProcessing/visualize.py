import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataProcessing.transform import get_train_transforms
from src.dataProcessing.augmentation_config import augmentation_profiles


def interactive_visualize(image_dir, mask_dir):
    """
    Åpner ett vindu der du kan bla i bilde- og maskepar med piltaster.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("Trykk ⬅️ eller ➡️ for å bla")
    idx = [0]

    def show(i):
        image_path = os.path.join(image_dir, image_files[i])
        mask_path = os.path.join(mask_dir, image_files[i].replace("image", "mask"))
        img = Image.open(image_path)
        mask = Image.open(mask_path)

        ax[0].imshow(img)
        ax[0].set_title(f"Bilde: {image_files[i]}")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Maske")
        for a in ax:
            a.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            idx[0] = (idx[0] + 1) % len(image_files)
            show(idx[0])
        elif event.key == "left":
            idx[0] = (idx[0] - 1) % len(image_files)
            show(idx[0])
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show(idx[0])
    plt.tight_layout()
    plt.show()


def visualize_multiple_augmentations(image_path, mask_path, cfg_name="basic", n=4):
    """Visualiserer flere augmentasjoner av et bilde og tilhørende maske."""
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
        cfg_name="default",  # Se augmentation_config.py
    )
