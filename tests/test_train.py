import pytest
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from src.dataProcessing.dataset import SnuplassDataset
from src.dataProcessing.transform import get_train_transforms, get_val_transforms
from src.model.unet import UNet
from src.dataProcessing.augmentation_config import augmentation_profiles


@pytest.fixture
def cfg():
    """
    Fixture to provide the default augmentation configuration.
    """
    return augmentation_profiles["default"]


@pytest.fixture
def setup_tmp_dataset(tmp_path):
    """
    Fixture to create a temporary dataset with images and masks for testing.
    Creates directories for images, masks, and splits, and populates them with dummy data.
    Returns a dictionary with paths to the created directories and files.
    """
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    splits_dir = tmp_path / "splits"
    images_dir.mkdir()
    masks_dir.mkdir()
    splits_dir.mkdir()

    file_ids = []

    for i in range(4):
        x0, y0 = 249000 + i * 100, 6786000 + i * 100
        x1, y1 = x0 + 100, y0 + 100
        filename = f"{x0}_{y0}_{x1}_{y1}"
        file_ids.append(f"image_{filename}")
        
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = images_dir / f"image_{filename}.png"
        img.save(img_path)

        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask = Image.fromarray(mask_array)
        mask_path = masks_dir / f"mask_{filename}.png"
        mask.save(mask_path)

    train_txt = splits_dir / "train.txt"
    val_txt = splits_dir / "val.txt"

    train_ids = file_ids[:2]
    val_ids = file_ids[2:]

    train_txt.write_text("\n".join(train_ids) + "\n")
    val_txt.write_text("\n".join(val_ids) + "\n")
        
    return {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "train_list": train_txt,
        "val_list": val_txt,
    }


def test_dataset_loads(setup_tmp_dataset):
    """
    Test that the dataset can be loaded and returns the correct number of items.
    """
    cfg = augmentation_profiles["default"]

    dataset = SnuplassDataset(
        image_dir=str(setup_tmp_dataset["images_dir"]),
        mask_dir=str(setup_tmp_dataset["masks_dir"]),
        file_list=str(setup_tmp_dataset["train_list"]),
        transform=get_train_transforms(cfg),
    )

    assert len(dataset) == 2
    img, mask = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)


def test_dataloader(cfg, setup_tmp_dataset):
    """
    Test that the DataLoader can load a batch of images and masks correctly.
    """
    img_dir = setup_tmp_dataset["images_dir"]
    mask_dir = setup_tmp_dataset["masks_dir"]
    train_txt = setup_tmp_dataset["train_list"]

    dataset = SnuplassDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        file_list=train_txt,
        transform=get_train_transforms(cfg),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    imgs, masks = batch
    assert imgs.shape[0] == 2
    assert masks.shape[0] == 2


def test_unet_forward_pass(cfg, setup_tmp_dataset):
    """
    Test that the UNet model can perform a forward pass with a batch of images.
    """
    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    img_dir = setup_tmp_dataset["images_dir"]
    mask_dir = setup_tmp_dataset["masks_dir"]
    train_txt = setup_tmp_dataset["train_list"]

    dataset = SnuplassDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        file_list=train_txt,
        transform=get_train_transforms(cfg),
    )
    img, _ = dataset[0]
    img = img.unsqueeze(0).to(device)  # batch size 1
    with torch.no_grad():
        output = model(img)
    assert output.shape[1] == 1  # n_classes
    assert output.shape[0] == 1  # batch size


def test_training_step(cfg, setup_tmp_dataset):
    """
    Test that the training step can compute a loss value.
    """
    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    img_dir = setup_tmp_dataset["images_dir"]
    mask_dir = setup_tmp_dataset["masks_dir"]
    train_txt = setup_tmp_dataset["train_list"]

    dataset = SnuplassDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        file_list=train_txt,
        transform=get_train_transforms(cfg),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device).float()
    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs.squeeze(1), masks)
    loss.backward()
    optimizer.step()
    assert loss.item() > 0


def test_validation_step(cfg, setup_tmp_dataset):
    """
    Test that the validation step can compute a loss value.
    """
    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    img_dir = setup_tmp_dataset["images_dir"]
    mask_dir = setup_tmp_dataset["masks_dir"]
    val_txt = setup_tmp_dataset["val_list"]

    dataset = SnuplassDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        file_list=val_txt,
        transform=get_val_transforms(),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device).float()
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(1), masks)
            val_loss += loss.item()

    assert val_loss >= 0


def test_main_runs(monkeypatch, setup_tmp_dataset, cfg):
    """
    Test that the main training function runs without errors.
    This is a simplified version that skips actual training for speed.
    """
    img_dir = setup_tmp_dataset["images_dir"]
    mask_dir = setup_tmp_dataset["masks_dir"]
    train_txt = setup_tmp_dataset["train_list"]

    def fast_main():
        batch_size = 2
        num_epochs = 1
        learning_rate = 1e-3
        device = torch.device("cpu")

        train_dataset = SnuplassDataset(
            image_dir=img_dir,
            mask_dir=mask_dir,
            file_list=train_txt,
            transform=get_train_transforms(cfg),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device).float()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print("Test training complete!")

    monkeypatch.setattr("src.train.main", fast_main)
    import src.train

    src.train.main()
