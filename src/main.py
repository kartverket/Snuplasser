from lightning_model import DCSwin
from loader import get_dataloader
import lightning as L

def parse_args():
    pass

if __name__ == "__main__":

    starting_point = [250000.0000, 6796000.0000]
    ending_point = [255000.0000, 6799000.0000]
    #starting_point = [250700.0000, 6796000.0000]
    #ending_point = [251700.0000, 6797000.0000]
    preferred_image_size = [500, 500]
    resolution = 0.2
    delay = 0.1  # Seconds to wait between each request

    bbox_size = [preferred_image_size[0] * resolution, preferred_image_size[1] * resolution]

    data_folder = f"{starting_point[0]}_{starting_point[1]}_{ending_point[0]}_{ending_point[1]}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}"
    #image_folder = data_folder + "/images"

    train_loader, num_classes = get_dataloader(data_folder, "train", 8, 0.8)
    val_loader, num_classes = get_dataloader(data_folder, "val", 1, 0.2)
    print(len(train_loader), len(val_loader))
    
    learning_rate = 1e-2
    
    model_size = "base"

    lightning_model = DCSwin(num_classes, learning_rate, train_loader, val_loader, model_size=model_size)

    model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{model_size}_"+ "{epoch}-{val_iou:.2f}",
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    trainer = L.Trainer(max_epochs=100, callbacks=[model_checkpoint])

    trainer.fit(lightning_model, train_loader, val_loader)


