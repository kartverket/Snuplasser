from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
import matplotlib.pyplot as plt
import mlflow
import torch
import warnings

def get_early_stopping(config):
    return EarlyStopping(
        monitor=config.get("monitor", "val_loss"),  # val_IoU
        mode=config.get("monitor_mode", "min"),     # "max" for IoU
        patience=config.get("early_stopping_patience", 5),
        verbose=True
    )

def get_model_checkpoint(config):
    metric_name = config.get("monitor", "val_loss")  # val_IoU
    filename = f"{{epoch:02d}}-{{{metric_name}:.4f}}"
    return ModelCheckpoint(
        monitor=metric_name,  
        mode=config.get("monitor_mode", "min"),     # "max" for IoU
        save_top_k=1,
        save_weights_only=True,
        filename=filename,       
    )


class LogPredictionsCallback(Callback):
    def __init__(self, log_every_n_epochs=5, artifact_dir="val_predictions"):
        self.log_every_n_epochs = log_every_n_epochs
        self.artifact_dir = artifact_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        x, y = batch
        x, y = x.to(pl_module.device), y.to(pl_module.device)
        logits = pl_module(x)
        preds = torch.sigmoid(logits) > 0.5

        num_images = 20
        max_images = min(num_images, x.shape[0])

        for i in range(max_images):
            image = x[i, :3].detach().cpu()    # RGB
            dom = x[i, 3].detach().cpu()       # DOM-kanal (må bruke indeks [0, 3])
            target = y[i].detach().cpu()
            pred = preds[i].detach().cpu()

            fig, axs = plt.subplots(1, 4, figsize=(14, 4))
            axs[0].imshow(image.permute(1, 2, 0))
            axs[0].set_title("Input RGB")
            axs[1].imshow(dom, cmap="gray")
            axs[1].set_title("Input DOM")
            axs[2].imshow(target.squeeze(), cmap="gray")
            axs[2].set_title("Target mask")
            axs[3].imshow(pred.squeeze(), cmap="gray")
            axs[3].set_title("Predicted mask")
            for ax in axs:
                ax.axis("off")
            fig.tight_layout()

            image_dir = os.path.join(self.artifact_dir, f"image_{i}")
            os.makedirs(image_dir, exist_ok=True)

            fname = os.path.join(image_dir, f"epoch_{trainer.current_epoch}.png")
            fig.savefig(fname)
            plt.close(fig)

            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            mlflow_client.log_artifact(run_id, fname, artifact_path=image_dir)

            os.remove(fname)