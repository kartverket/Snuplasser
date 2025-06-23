import lightning as L

import src.config as config
from model.lightning_model import DCSwin
from dataProcessing.loader import get_dataloader

# For å splitte datasettet med strukturen fra dataset.py
# from dataProcessing.dataset import load_numpy_split_stack

# image_dir = "data/images"
# mask_dir = "data/masks"

# (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_numpy_split_stack(image_dir, mask_dir)



def parse_args():
    pass


if __name__ == "__main__":

    bbox = config.TEST_BBOX
    preferred_image_size = [500, 500]  # Bredde, Høyde i piksler
    resolution = 0.2  # Oppløsning i meter per piksel
    delay = 0.1  # Sekunder å vente mellom hver forespørsel

    data_folder = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}"

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
