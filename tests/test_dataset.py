import pytest
from PIL import Image
import numpy as np
import torch
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataProcessing.dataset import SnuplassDataset


def create_dummy_png(path, size=(256, 256), color=128, mode="L"):
    """Opprett en dummy PNG-fil i gråskala eller RGB."""
    arr = np.full(size + ((3,) if mode == "RGB" else ()), color, dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture
def dummy_dataset(tmp_path):
    """Opprett et dummy datasett med bilde- og maskefiler."""
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    split_dir = tmp_path / "splits"
    image_dir.mkdir()
    mask_dir.mkdir()
    split_dir.mkdir()

    # Opprett bilder og masker
    for i in range(2):
        create_dummy_png(image_dir / f"image_{i}.png", color=100, mode="RGB")
        create_dummy_png(
            mask_dir / f"mask_{i}.png", color=(255 if i == 0 else 0), mode="L"
        )

    # Opprett file_list.txt
    file_list = split_dir / "file_list.txt"
    with open(file_list, "w") as f:
        for i in range(2):
            f.write(f"image_{i}\n")

    dataset = SnuplassDataset(
        image_dir=str(image_dir), mask_dir=str(mask_dir), file_list=str(file_list)
    )
    return dataset


def test_dataset_length(dummy_dataset):
    assert len(dummy_dataset) == 2


def test_dataset_getitem_shape_and_type(dummy_dataset):
    image, mask = dummy_dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert mask.dtype == torch.float32
    assert mask.max() <= 1
    assert mask.min() >= 0


def test_create_train_val_split(tmp_path):
    # Opprett dummy bilder
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(10):
        create_dummy_png(image_dir / f"image_{i}.png", color=100, mode="RGB")

    # Kjør split
    split_dir = tmp_path / "splits"
    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    SnuplassDataset.create_train_val_split(
        image_dir=str(image_dir),
        train_file=str(train_file),
        val_file=str(val_file),
        split_ratio=0.7,
        seed=123,
    )

    # Verifiser innhold
    with open(train_file) as f:
        train_ids = [line.strip() for line in f]
    with open(val_file) as f:
        val_ids = [line.strip() for line in f]

    assert len(train_ids) == 7
    assert len(val_ids) == 3
    assert all(id_.startswith("image_") for id_ in train_ids + val_ids)

    # Verifiser metadatafil
    meta_path = split_dir / "split_meta.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["total_files"] == 10
    assert meta["num_train"] == 7
    assert meta["num_val"] == 3
    assert meta["split_ratio"] == 0.7
    assert meta["seed"] == 123
    assert "created" in meta
