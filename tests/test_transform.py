import pytest
import numpy as np
import torch

from src.dataProcessing.transform import get_train_transforms, get_val_transforms


@pytest.fixture
def dummy_image_and_mask():
    """
    Oppretter et dummy bilde og en dummy maske for testing.
    Returnerer et tuple med bilde og maske.
    """
    image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=(256, 256), dtype=np.uint8)
    return image, mask


def test_train_transforms_output(dummy_image_and_mask):
    """
    Tester at treningstransformasjoner returnerer riktige typer og formater.
    """
    image, mask = dummy_image_and_mask

    cfg = {
        "flip_p": 1.0,
        "rot90_p": 1.0,
        "brightness_p": 1.0,
    }

    transforms = get_train_transforms(cfg)
    result = transforms(image=image, mask=mask)

    transformed_image = result["image"]
    transformed_mask = result["mask"]

    assert isinstance(transformed_image, torch.Tensor)
    assert isinstance(transformed_mask, torch.Tensor)
    assert transformed_image.shape[0] == 3  # CHW format
    assert transformed_mask.ndim == 2
    assert transformed_image.dtype == torch.float32
    assert transformed_mask.dtype in [torch.int64, torch.uint8]


def test_val_transforms_output(dummy_image_and_mask):
    """
    Tester at valideringstransformasjoner returnerer riktige typer og formater.
    """
    image, mask = dummy_image_and_mask

    transforms = get_val_transforms()
    result = transforms(image=image, mask=mask)

    transformed_image = result["image"]
    transformed_mask = result["mask"]

    assert isinstance(transformed_image, torch.Tensor)
    assert isinstance(transformed_mask, torch.Tensor)
    assert transformed_image.shape[0] == 3
    assert transformed_mask.shape == mask.shape
    assert transformed_image.dtype == torch.float32
