import torch
from lightning.pytorch import LightningModule
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
from torchmetrics.segmentation import DiceScore
from model.losses.losses import DiceBCELoss
from model.losses.loss_utils import compute_loss_weights


class DeepLabV3Plus(LightningModule):
    """
    DeepLabV3+ med Pytorch Lightning wrapper.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = smp.DeepLabV3Plus(
            encoder_name=config.get("backbone", "resnet101"),
            encoder_weights=None,
            in_channels=config.get("in_channels", 4),
            classes=1,
        )

        self.lr = config.get("lr", 1e-3)

        maks_dir = config.get("data", {}).get("maks_dir")
        if maks_dir:
            dice_w, bce_w, pos_w = compute_loss_weights(maks_dir)
        else:
            dice_w, bce_w, pos_w = 0.5, 0.5, 1.0

        self.loss_fn = DiceBCELoss(
            dice_weight=dice_w, bce_weight=bce_w, pos_weight=pos_w
        )

        self.iou_metric = BinaryJaccardIndex()
        self.dice_metric = DiceScore(num_classes=1)
        self.accuracy_metric = BinaryAccuracy()

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255
        return self.model(x)

    def training_step(self, batch, _):
        x, y, _ = batch
        y = y.float()
        logts = self(x)
        loss = self.loss_fn(logts, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        with torch.no_grad():
            x, y, _ = batch
            y = y.float()
            logts = self(x)
            loss = self.loss_fn(logts, y)
            self.log("val_loss", loss, prog_bar=True)

            preds = torch.sigmoid(logts)
            pred_bin = (preds > 0.5).float()
            iou = self.iou_metric(pred_bin, y)
            dice = self.dice_metric(pred_bin, y)
            acc = self.accuracy_metric(pred_bin, y)

            self.log("val_iou", iou, prog_bar=True)
            self.log("val_dice", dice, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y, _ = batch
            y = y.float()
            logts = self(x)
            loss = self.loss_fn(logts, y)
            self.log("test_loss", loss, prog_bar=True)

            preds = torch.sigmoid(logts)
            pred_bin = (preds > 0.5).float()
            iou = self.iou_metric(pred_bin, y)
            dice = self.dice_metric(pred_bin, y)
            acc = self.accuracy_metric(pred_bin, y)

            self.log("test_iou", iou, prog_bar=True)
            self.log("test_dice", dice, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Batch er (image_tensor, mask_tensor, filename)
        x, _, filename = batch

        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        return {"filename": filename, "mask": preds, "image": x}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def get_deeplabv3plus_lightning_model(config):
    """
    Returnerer en instans av DeepLabV3Plus-modellen.
    """
    return DeepLabV3Plus(config=config)