import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback


def get_early_stopping(config: str) -> EarlyStopping:
    """
    Returnerer et EarlyStopping objekt som overvåker en metrikk.
    Argumenter:
        config: Konfigurasjonsfilen
    Returnerer:
        EarlyStopping objekt
    """
    return EarlyStopping(
        monitor=config.get("monitor", "val_loss"),  # val_IoU
        mode=config.get("monitor_mode", "min"),  # "max" for IoU
        patience=config.get("early_stopping_patience", 5),
        verbose=True,
    )


def get_model_checkpoint(config: dict) -> ModelCheckpoint:
    """
    Lagrer checkpoint av modellen.
    Argumenter:
        config: Konfigurasjonsfilen
    Returnerer:
        ModelCheckpoint objekt
    """
    metric_name = config.get("monitor", "val_loss")  # val_IoU
    filename = "best"
    return ModelCheckpoint(
        monitor=metric_name,
        mode=config.get("monitor_mode", "min"),  # "max" for IoU
        save_top_k=1,
        save_weights_only=True,
        filename=filename,
    )


class LogPredictionsCallback(Callback):
    """
    Logger prediksjoner til MLFlow.
    """
    def __init__(
        self,
        log_every_n_epochs=1,
        artifact_dir="val_predictions",
        always_log_ids=None,
        max_random_logs=10,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.artifact_dir = artifact_dir
        self.always_log_ids = set(always_log_ids or [])
        self.max_random_logs = max_random_logs
        self.all_true = []
        self.all_pred = []
        self.val_true = []
        self.val_pred = []
        self.display_labels = ["annet", "snuplass"]

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_true.clear()
        self.val_pred.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        device = pl_module.device
        y = y.to(device).long()
        preds = (torch.sigmoid(pl_module(x.to(device))) > 0.5).long()
        for y_img, p_img in zip(y.cpu(), preds.cpu()):
            self.val_true.append(y_img)
            self.val_pred.append(p_img)

    def on_validation_epoch_end(self, trainer, pl_module):
        dataloader = trainer.datamodule.val_dataloader()
        device = pl_module.device
        epoch = trainer.current_epoch
        logged = set()

        y_true = [int((m.numpy() == 1).any()) for m in self.val_true]
        y_pred = [int((p.numpy() == 1).any()) for p in self.val_pred]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.display_labels
        )
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Val_confusion matrix (epoch {trainer.current_epoch})")
        plt.tight_layout()
        run_id = trainer.logger.run_id
        trainer.logger.experiment.log_figure(
            run_id=run_id,
            figure=fig,
            artifact_file=f"val_confusion_matrix_epoch_{trainer.current_epoch}.png"
        )
        plt.close(fig)

        for batch in dataloader:
            if len(batch) != 3:
                continue
            x, y, fnames = batch
            x, y = x.to(device), y.to(device)
            preds = torch.sigmoid(pl_module(x)) > 0.5
            for i, fname in enumerate(fnames):
                name = Path(fname).name
                log_this = name in self.always_log_ids or (
                    epoch % self.log_every_n_epochs == 0
                    and len(logged) < self.max_random_logs
                )
                if log_this and name not in logged:
                    self._log_prediction(x[i], y[i], preds[i], name, epoch, trainer)
                    logged.add(name)
            if len(logged) >= self.max_random_logs and not (
                self.always_log_ids - logged
            ):
                break

    def on_test_epoch_start(self, trainer, pl_module):
        self.all_true.clear()
        self.all_pred.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        device = pl_module.device
        y = y.to(device).long()
        preds = (torch.sigmoid(pl_module(x.to(device))) > 0.5).long()
        for y_img, p_img in zip(y.cpu(), preds.cpu()):
            self.all_true.append(y_img)
            self.all_pred.append(p_img)

    def on_test_epoch_end(self, trainer, pl_module):
        y_true = [int((m.numpy() == 1).any()) for m in self.all_true]
        y_pred = [int((p.numpy() == 1).any()) for p in self.all_pred]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.display_labels
        )
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion matrix (epoch {trainer.current_epoch})")
        plt.tight_layout()
        run_id = trainer.logger.run_id
        trainer.logger.experiment.log_figure(
            run_id=run_id,
            figure=fig,
            artifact_file=f"confusion_matrix_epoch_{trainer.current_epoch}.png"
        )
        plt.close(fig)

    def _log_prediction(self, x, y, pred, name, epoch, trainer):
        img, dom = x[:3], x[3:]
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np / 255.0
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
        axs[2].set_title("Fasit")
        axs[3].set_title("Prediksjon")
        plt.tight_layout()
        artifact_path = f"{self.artifact_dir}/{name}/epoch_{epoch}.png"
        trainer.logger.experiment.log_figure(
            run_id=trainer.logger.run_id, figure=fig, artifact_file=artifact_path
        )
        plt.close(fig)


def log_predictions_from_preds(
    preds: List[dict[str, float | str | torch.Tensor]],
    logger: "MLflowLogger",
    artifact_dir: str="predictions",
    local_save_dir: str="/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/predicted_masks",
    max_logs: int=20,
):
    """
    Logger prediksjoner både til MLFlow og til en lokal mappe.
    Argumenter:
        preds (List): en liste med prediksjoner
        logger (MLflowLogger): MLFlowLogger som skal brukes for logging
        artifact_dir (str): hvor prediksjoner skal logges til MLFlow
        local_save_dir (str): hvor prediksjoner skal lagres lokalt
        max_logs (int): maksimalt antall prediksjoner som skal logges til MLFlow
    """
    if not hasattr(logger, "run_id") or logger.run_id is None:
        raise RuntimeError("MLFlowLogger må ha en aktiv kjøring for å logge.")
    for batch in preds:
        filenames = batch.get("filename")
        masks = batch.get("mask")
        images = batch.get("image")
        for i in range(len(filenames)):
            rgb_tensor = images[i, :3] if images is not None else None
            dom_tensor = images[i, 3] if images is not None else None
            _log_prediction_artifact(
                rgb_tensor=rgb_tensor,
                dom_tensor=dom_tensor,
                pred_tensor=masks[i],
                fname=filenames[i],
                logger=logger,
                artifact_dir=artifact_dir,
                local_save_dir=local_save_dir,
            )


def _log_prediction_artifact(
    rgb_tensor: torch.Tensor, dom_tensor: torch.Tensor, pred_tensor: torch.Tensor, fname: str, logger: "MLflowLogger", artifact_dir: str, local_save_dir: str
):
    """
    Logger en enkelt prediksjon til MLFlow og til en lokal mappe.
    Argumenter:
        rgb_tensor (torch.Tensor): en tensor med RGB-bilder
        dom_tensor (torch.Tensor): en tensor med DOM-bilder
        pred_tensor (torch.Tensor): en tensor med prediksjoner
        fname (str): filnavnet på bildet
        logger (MLflowLogger): MLFlowLogger som skal brukes for logging
        artifact_dir (str): hvor prediksjoner skal logges til MLFlow
        local_save_dir (str): hvor prediksjoner skal lagres lokalt
    """
    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_np = rgb_np / 255.0 
    dom_np = dom_tensor.cpu().numpy()
    pred_np = pred_tensor.squeeze().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(rgb_np)
    axs[0].set_title("RGB")
    axs[1].imshow(dom_np, cmap="gray")
    axs[1].set_title("DOM")
    axs[2].imshow(pred_np, cmap="gray")
    axs[2].set_title("Prediksjon")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    artifact_path = f"{artifact_dir}/{Path(fname).stem}.png"
    logger.experiment.log_figure(
        run_id=logger.run_id, figure=fig, artifact_file=artifact_path
    )
    plt.close(fig)
    # Lagrer bare prediksjonen lokalt, ikke RGB og DOM
    fig_pred, ax_pred = plt.subplots()
    ax_pred.imshow(pred_np, cmap="gray")
    ax_pred.axis("off")
    os.makedirs(local_save_dir, exist_ok=True)
    local_path = os.path.join(local_save_dir, f"pred_{Path(fname).stem}.png")
    fig_pred.savefig(local_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig_pred)