import pytest
from PIL import Image
import numpy as np
import torch
from src.dataProcessing.dataset import SnuplassDataset


def create_dummy_png(path, size=(256, 256), color=128):
    """
    Lag en dummy PNG-fil med en ensfarget bakgrunn.
    Args:
        path (str): Stien der PNG-filen skal lagres.
        size (tuple): Størrelsen på bildet (bredde, høyde).
        color (int): Gråtonen for bildet (0-255).
    """
    Image.fromarray(np.full(size, color, dtype=np.uint8)).save(path)


@pytest.fixture
def dummy_dataset(tmp_path):
    """
    Opprett et dummy datasett for testing.
    Oppretter en mappe med bilder og masker, samt en fil som lister opp
    hvilke filer som skal brukes.
    """
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    file_list_path = tmp_path / "file_list.txt"

    # Lag dummy filer
    filenames = ["image_1", "image_2"]
    with open(file_list_path, "w") as f:
        for name in filenames:
            f.write(name + "\n")
            create_dummy_png(image_dir / f"{name}.png", color=100)
            create_dummy_png(mask_dir / f"mask_1.png", color=255)
            create_dummy_png(mask_dir / f"mask_2.png", color=0)

    dataset = SnuplassDataset(str(image_dir), str(mask_dir), str(file_list_path))
    return dataset


def test_dataset_length(dummy_dataset):
    """
    Tester at datasettet har riktig lengde basert på filene i file_list.txt.
    """
    assert len(dummy_dataset) == 2


def test_dataset_getitem_shape_and_type(dummy_dataset):
    """
    Tester at __getitem__ returnerer bilder og masker med riktig form og type.
    """
    image, mask = dummy_dataset[0]
    assert isinstance(image, np.ndarray)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert mask.dtype == torch.long
    assert mask.max() <= 1
    assert mask.min() >= 0
