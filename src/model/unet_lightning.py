import torch
from torch import nn
from lightning.pytorch import LightningModule
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
from torchmetrics.segmentation import DiceScore


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.dice(preds, targets) + self.bce(preds, targets)


class UNetLightning(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=config.get("encoder", "resnet18"),
            encoder_weights=None,
            in_channels=config.get("in_channels", 4),
            classes=1,  # én kanal for binær segmentering
        )
        self.lr = config.get("lr", 1e-3)

        self.loss_fn = DiceBCELoss()
        #self.loss_fn = nn.BCEWithLogitsLoss()

        self.iou_metric = BinaryJaccardIndex()
        self.dice = DiceScore(num_classes=2)
        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()  # nødvendig for BCEWithLogits
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)

        preds = torch.sigmoid(logits)
        pred_bin = (preds > 0.5).float()

        iou = self.iou_metric(preds, y)
        dice = self.dice(preds, y)
        acc = self.accuracy(pred_bin, y)

        self.log("val_iou", iou, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_unet_lightning(config):
    return UNetLightning(config)
