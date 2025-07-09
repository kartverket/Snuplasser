from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
import matplotlib.pyplot as plt
import mlflow
import torch
import warnings
from pathlib import Path

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
    def __init__(self, log_every_n_epochs=1, artifact_dir="val_predictions", always_log_ids=None, max_random_logs=10):
        self.log_every_n_epochs = log_every_n_epochs
        self.artifact_dir = artifact_dir
        self.always_log_ids = set(always_log_ids or [])
        self.max_random_logs = max_random_logs

    def on_validation_epoch_end(self, trainer, pl_module):
        dataloader = trainer.datamodule.val_dataloader()
        device = pl_module.device
        epoch = trainer.current_epoch
        logged = set()

        for batch in dataloader:
            if len(batch) != 3:
                continue
            x, y, fnames = batch
            x, y = x.to(device), y.to(device)
            preds = torch.sigmoid(pl_module(x)) > 0.5

            for i, fname in enumerate(fnames):
                name = Path(fname).name
                log_this = (
                    name in self.always_log_ids or
                    (epoch % self.log_every_n_epochs == 0 and len(logged) < self.max_random_logs)
                )
                if log_this and name not in logged:
                    self._log_prediction(x[i], y[i], preds[i], name, epoch, trainer)
                    logged.add(name)

            if len(logged) >= self.max_random_logs and not (self.always_log_ids - logged):
                break

    def _log_prediction(self, x, y, pred, name, epoch, trainer):
        img, dom = x[:3], x[3:]
    
        # Sørg for 2D-tensorer til visning
        img_np = img.permute(1, 2, 0).cpu().numpy()
        dom_np = dom.squeeze().cpu().numpy()
        y_np = y.squeeze().cpu().numpy()
        pred_np = pred.squeeze().cpu().numpy()

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].imshow(img_np)
        axs[1].imshow(dom_np, cmap="gray")
        axs[2].imshow(y_np, cmap="gray")
        axs[3].imshow(pred_np, cmap="gray")

        for ax in axs:
            ax.axis("off")

        axs[0].set_title("Input RGB")
        axs[1].set_title("Input DOM")
        axs[2].set_title("Target mask")
        axs[3].set_title("Predicted mask")

        plt.tight_layout()
        artifact_path = f"{self.artifact_dir}/image_{name}/epoch_{epoch}.png"
        trainer.logger.experiment.log_figure(
            run_id=trainer.logger.run_id,
            figure=fig,
            artifact_file=artifact_path
        )
        plt.close(fig)