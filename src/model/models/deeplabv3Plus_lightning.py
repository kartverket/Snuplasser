import torch
from lightning.pytorch import LightningModule
import segmentation_models_pytorch as smp
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryAccuracy,
    BinaryF1Score,
)
from ..losses.losses import DiceBCELoss
from ..losses.loss_utils import compute_loss_weights


class DeepLabV3Plus(LightningModule):
    """
    DeepLabV3+ med Pytorch Lightning wrapper.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        model_name = config.get("model_names", [])[0]
        model_cfg = config.get("model", {}).get(model_name, {})

        self.model = smp.DeepLabV3Plus(
            encoder_name=config.get("backbone", "resnet50"),
            encoder_weights=None,
            in_channels=config.get("in_channels", 4),
            classes=1,
        )

        self.lr = model_cfg.get("lr", 1e-3)
        self.wd = model_cfg.get("wd", 1e-5)

        mask_dir = config.get("data", {}).get("mask_dir")

        log_pred_cfg = config.get("log_predictions_callback", {})
        self.threshold = log_pred_cfg.get("threshold", 0.6)

        training_cfg = config.get("training", {})
        self.scheduler_factor = training_cfg.get("scheduler_factor", 0.1)
        self.scheduler_patience = training_cfg.get("scheduler_patience", 5)

        if mask_dir:
            dice_w, bce_w, pos_w = compute_loss_weights(mask_dir)
        else:
            dice_w, bce_w, pos_w = 0.7, 0.3, 1.0

        self.loss_fn = DiceBCELoss(
            dice_weight=dice_w, bce_weight=bce_w, pos_weight=pos_w
        )

        self.iou_metric = BinaryJaccardIndex()
        self.dice_metric = BinaryF1Score()
        self.accuracy_metric = BinaryAccuracy()

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255
        return self.model(x)

    def training_step(self, batch, _):
        x, y, _ = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, _):
        x, y, _ = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x.shape[0])

        preds = torch.sigmoid(logits)
        pred_bin = (preds > self.threshold).float()

        iou = self.iou_metric(preds, y)
        dice = self.dice_metric(preds, y)
        acc = self.accuracy_metric(pred_bin, y)

        self.log("val_iou", iou, prog_bar=True, batch_size=x.shape[0])
        self.log("val_dice", dice, prog_bar=True, batch_size=x.shape[0])

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            y = y.float()
            logts = self(x)
            loss = self.loss_fn(logts, y)
            self.log("test_loss", loss, prog_bar=True, batch_size=x.shape[0])

            preds = torch.sigmoid(logts)
            pred_bin = (preds > self.threshold).float()
            iou = self.iou_metric(pred_bin, y)
            dice = self.dice_metric(pred_bin, y)
            acc = self.accuracy_metric(pred_bin, y)
            self.log("test_iou", iou, prog_bar=True, batch_size=x.shape[0])
            self.log("test_dice", dice, prog_bar=True, batch_size=x.shape[0])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Batch er (image_tensor, mask_tensor, filename)
        x, _, filename = batch

        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()

        return {"filename": filename, "mask": preds, "image": x}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.get("monitor_mode", "max"),
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.get("monitor", "val_dice"),
                "interval": "epoch",
                "frequency": 1,
            },
        }


def get_deeplabv3plus_lightning_model(config):
    """
    Returnerer en instans av DeepLabV3Plus-modellen.
    """
    return DeepLabV3Plus(config=config)
