import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataProcessing.dataset import SnuplassDataset
from src.dataProcessing.transform import get_train_transforms, get_val_transforms
from src.model.unet import UNet
from src.dataProcessing.augmentation_config import augmentation_profiles


def main():
    cfg = augmentation_profiles["default"]
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        dom_dir="data/doms",
        file_list="data/splits/train.txt",
        transform=get_train_transforms(cfg),
    )

    val_dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        dom_dir="data/doms",
        file_list="data/splits/val.txt",
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(n_channels=4, n_classes=1, bilinear=False).to(
        device
    )  # bare å bytte modell

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            images, masks = images.to(device), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nTrain loss: {avg_train_loss:.4f}")

        # Validering
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                images, masks = images.to(device), masks.to(device).float()
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Val loss: {avg_val_loss:.4f}")

    print("✅ Trening ferdig")


if __name__ == "__main__":
    main()
