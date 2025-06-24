import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def export_to_hdf5(image_dir, dom_dir, mask_dir, output_path):
    """
    Kombinerer RGB-bilder med DOM og tilhørende maskefiler,
    og lagrer dem som HDF5-format for enkel opplasting til f.eks. databricks.
    """
    image_files = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.startswith("image_") and f.endswith(".png")
        ]
    )
    valid_ids = []

    for f in image_files:
        suffix = f[len("image_") :]
        dom_path = os.path.join(dom_dir, f"dom_{suffix}")
        mask_path = os.path.join(mask_dir, f"mask_{suffix}")
        image_path = os.path.join(image_dir, f)

        if os.path.exists(dom_path) and os.path.exists(mask_path):
            valid_ids.append(suffix)
        else:
            print(f"⚠️ Mangler filer for: {suffix}")
            if not os.path.exists(image_path):
                print(f"  - Mangler bilde: {image_path}")
            if not os.path.exists(dom_path):
                print(f"  - Mangler DOM: {dom_path}")
            if not os.path.exists(mask_path):
                print(f"  - Mangler maske: {mask_path}")

    if not valid_ids:
        raise ValueError(
            f"Ingen gyldige bildepar funnet.\n"
            f"Sjekk at alle tre filer finnes for hvert ID i {image_dir}, {dom_dir} og {mask_dir}."
        )

    sample_image = Image.open(os.path.join(image_dir, f"image_{valid_ids[0]}"))
    width, height = sample_image.size

    num_samples = len(valid_ids)
    image_stack = np.zeros((num_samples, height, width, 3), dtype=np.uint8)
    dom_stack = np.zeros((num_samples, height, width, 1), dtype=np.uint8)
    mask_stack = np.zeros((num_samples, height, width, 1), dtype=np.uint8)

    print(f"📦 Eksporterer {num_samples} eksempler til {output_path}")
    for i, suffix in enumerate(tqdm(valid_ids)):
        image = np.array(Image.open(os.path.join(image_dir, f"image_{suffix}")))
        dom = np.array(Image.open(os.path.join(dom_dir, f"dom_{suffix}")))
        mask = np.array(Image.open(os.path.join(mask_dir, f"mask_{suffix}")))

        image_stack[i] = image
        dom_stack[i, ..., 0] = dom
        mask_stack[i, ..., 0] = mask // 255  # binariser

    stacked_input = np.concatenate([image_stack, dom_stack], axis=-1)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("images_dom", data=stacked_input, compression="gzip")
        hf.create_dataset("masks", data=mask_stack, compression="gzip")

    print(f"✅ HDF5-fil lagret: {output_path}")


def print_h5_structure(file_path):
    def recurse(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"📦 Dataset: {name} | shape: {obj.shape} | dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"📁 Group: {name}")

    with h5py.File(file_path, "r") as f:
        f.visititems(recurse)


if __name__ == "__main__":
    export_to_hdf5(
        image_dir="data/images",
        dom_dir="data/doms",
        mask_dir="data/masks",
        output_path="data/combined_dataset.h5",
    )
    print_h5_structure("data/combined_dataset.h5")
