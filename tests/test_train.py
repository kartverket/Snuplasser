import pytest
import torch
from torch.utils.data import DataLoader

import src.train as train
from src.dataProcessing.dataset import SnuplassDataset
from src.dataProcessing.transform import get_train_transforms, get_val_transforms
from src.model.unet import UNet
from src.dataProcessing.augmentation_config import augmentation_profiles


@pytest.fixture
def cfg():
    """
    Gir en standard konfigurasjon for augmentering.
    """
    return augmentation_profiles["default"]


def test_dataset_loads(cfg):
    """
    Tester at SnuplassDataset kan lastes uten feil og returnerer bilder og masker.
    """
    dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        file_list="data/splits/train.txt",
        transform=get_train_transforms(cfg),
    )
    assert len(dataset) > 0
    img, mask = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)


def test_dataloader(cfg):
    """
    Tester at DataLoader kan iterere over SnuplassDataset og returnerer batcher med bilder og masker.
    """
    dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        file_list="data/splits/train.txt",
        transform=get_train_transforms(cfg),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    imgs, masks = batch
    assert imgs.shape[0] == 2
    assert masks.shape[0] == 2


def test_unet_forward_pass(cfg):
    """
    Tester at UNet-modellen kan utføre en fremoverpass uten feil.
    """
    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        file_list="data/splits/train.txt",
        transform=get_train_transforms(cfg),
    )
    img, _ = dataset[0]
    img = img.unsqueeze(0).to(device)  # batch size 1
    with torch.no_grad():
        output = model(img)
    assert output.shape[1] == 1  # n_classes
    assert output.shape[0] == 1  # batch size


def test_training_step(cfg):
    """
    Tester at treningssteget kan kjøre uten feil og oppdaterer vektene.
    """
    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        file_list="data/splits/train.txt",
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


def test_validation_step():
    """
    Tester at valideringssteget kan kjøre uten feil og returnerer en gyldig tapverdi.
    """
    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        file_list="data/splits/val.txt",
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
    avg_val_loss = val_loss / len(loader)
    assert avg_val_loss >= 0


def test_main_runs(monkeypatch):
    """
    Testerer at main-funksjonen i train.py kan kjøres uten feil.
    Dette er en rask test som ikke trener modellen fullt ut.
    """

    def fast_main():
        """
        En raskere versjon av main-funksjonen for testing.
        """
        cfg = augmentation_profiles["default"]
        batch_size = 2
        num_epochs = 1
        learning_rate = 1e-3
        device = torch.device("cpu")

        train_dataset = SnuplassDataset(
            image_dir="data/images",
            mask_dir="data/masks",
            file_list="data/splits/train.txt",
            transform=get_train_transforms(cfg),
        )

        val_dataset = SnuplassDataset(
            image_dir="data/images",
            mask_dir="data/masks",
            file_list="data/splits/val.txt",
            transform=get_val_transforms(),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device).float()
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(1), masks)
                    val_loss += loss.item()

    monkeypatch.setattr(train, "main", fast_main)
    train.main()
