import lightning as L

import src.config as config
from model.lightning_model import DCSwin
from dataProcessing.loader import get_dataloader


def parse_args():
    pass


if __name__ == "__main__":

    starting_point = config.STARTING_POINT
    ending_point = config.ENDING_POINT
    preferred_image_size = [500, 500]  # Bredde, Høyde i piksler
    resolution = 0.2  # Oppløsning i meter per piksel
    delay = 0.1  # Sekunder å vente mellom hver forespørsel

    bbox_size = [
        preferred_image_size[0] * resolution,
        preferred_image_size[1] * resolution,
    ]

    data_folder = f"{starting_point[0]}_{starting_point[1]}_{ending_point[0]}_{ending_point[1]}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}"

    train_loader, num_classes = get_dataloader(data_folder, "train", 8, 0.8)
    val_loader, num_classes = get_dataloader(data_folder, "val", 1, 0.2)

    learning_rate = 1e-2

    model_size = "base"

    lightning_model = DCSwin(
        num_classes, learning_rate, train_loader, val_loader, model_size=model_size
    )

    model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{model_size}_" + "{epoch}-{val_iou:.2f}",
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = L.Trainer(max_epochs=100, callbacks=[model_checkpoint])

    trainer.fit(lightning_model, train_loader, val_loader)
