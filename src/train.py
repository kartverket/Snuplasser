import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataProcessing.dataset import SnuplassDataset
from src.dataProcessing.transform import get_train_transforms, get_val_transforms
from src.model.unet import UNet
from src.dataProcessing.augmentation_config import augmentation_profiles
from src.utils import iou_pytorch, acc_pytorch


def main():
    cfg = augmentation_profiles["default"]
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = "runs/snuplasser"  # "/dbfs/tmp/tensorboard_logs" for Databricks
    writer = SummaryWriter(log_dir=log_dir)

    train_dataset = SnuplassDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        dom_dir="data/doms",
        file_list="data/splits/train.txt",
        transform=get_train_transforms(cfg, ratio=None),  # ratio=None for baseline
        # For å bruke augmentering, sett ratio til en verdi mellom 0 og 1
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
        # Trening
        model.train()
        total_loss = 0

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            images, masks = images.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nTrain loss: {avg_train_loss:.4f}")

        writer.add_scalar("Tap/Trening", avg_train_loss, epoch)

        # Validering
        model.eval()
        val_loss = 0.0
        val_ious = []
        val_accs = []
        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                images, masks = images.to(device).float(), masks.to(device).float()
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), masks)
                val_loss += loss.item()

                # Beregn IoU og accuracy
                predictions = (
                    torch.sigmoid(outputs) > 0.5
                ).int()  # Konverterer til binære prediksjoner
                iou = iou_pytorch(predictions, masks.int())
                acc = acc_pytorch(predictions, masks.int())
                val_ious.append(iou.item())
                val_accs.append(acc.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Val loss: {avg_val_loss:.4f}")
        avg_iou = sum(val_ious) / len(val_ious)
        avg_acc = sum(val_accs) / len(val_accs)

        writer.add_scalar("Tap/Validering", avg_val_loss, epoch)
        writer.add_scalar("Metrikker/IoU", avg_iou, epoch)
        writer.add_scalar("Metrikker/Accuracy", avg_acc, epoch)

    writer.close()

    print("✅ Trening ferdig")


if __name__ == "__main__":
    main()
