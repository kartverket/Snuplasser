import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataProcessing.dataset import SnuplassDataset
from dataProcessing.transform import get_train_transforms, get_val_transforms
from model.unet import UNet
from dataProcessing.augmentation_config import augmentation_profiles


def main():
    cfg = augmentation_profiles["default"]
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(
        device
    )  # bare å bytte modell

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}"
        )

    print("Training complete!")


if __name__ == "__main__":
    main()
