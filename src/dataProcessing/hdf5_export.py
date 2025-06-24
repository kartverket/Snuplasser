
import os
import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm

def export_to_hdf5(image_dir, dom_dir, mask_dir, output_path, image_size=(512, 512)):
    """
    Kombinerer RGB-bilder med DOM og tilhørende maskefiler,
    og lagrer dem som HDF5-format for enkel opplasting til f.eks. databricks.
    """
    inputs, masks = [], []
    image_files = sorted([f for f in os.listdir(image_dir) if f.startswith("image_") and f.endswith(".png")])

    for file in tqdm(image_files, desc="🔄 Behandler bilder"):
        stem = file.replace("image_", "").replace(".png", "")
        image_path = os.path.join(image_dir, file)
        dom_path = os.path.join(dom_dir, f"dom_{stem}.png")
        mask_path = os.path.join(mask_dir, f"mask_{stem}.png")

        if not os.path.exists(dom_path) or not os.path.exists(mask_path):
            print(f"⚠️ Mangler dom eller mask for {stem}")
            continue

        # Last inn og skaler alle filer til samme størrelse
        image = Image.open(image_path).resize(image_size)
        dom = Image.open(dom_path).resize(image_size)
        mask = Image.open(mask_path).resize(image_size)

        # Konverter til numpy
        image_np = np.array(image)
        dom_np = np.array(dom)
        mask_np = np.array(mask) // 255  # Binariser

        # Ekstra sjekk: dom må være énkanals
        if dom_np.ndim == 3:
            dom_np = dom_np[:, :, 0]

        # Sjekk samsvar i form
        if image_np.shape[:2] != dom_np.shape[:2] or image_np.shape[:2] != mask_np.shape[:2]:
            print(f"⚠️ Ulik størrelse på bilde, dom eller mask for {stem} – hopper over")
            continue

        # Stack som [H, W, 4]
        combined = np.dstack((image_np, dom_np))
        inputs.append(combined)
        masks.append(mask_np)

    if not inputs:
        raise ValueError("Ingen gyldige bildepar funnet!")

    inputs = np.stack(inputs)
    masks = np.stack(masks)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("images", data=inputs, compression="gzip")
        f.create_dataset("masks", data=masks, compression="gzip")

    print(f"✅ Lagret {inputs.shape[0]} bilder til {output_path}")
    print(f"📐 Shape: inputs={inputs.shape}, masks={masks.shape}")


if __name__ == "__main__":
    export_to_hdf5(
        image_dir="data/images",
        dom_dir="data/dom",
        mask_dir="data/masks",
        output_path="data/combined_dataset.h5"
    )
