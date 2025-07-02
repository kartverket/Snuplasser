import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F

class UNetLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = smp.Unet(
            encoder_name=config.get("encoder", "resnet50"),
            encoder_weights=config.get("encoder_weights", "imagenet"),
            in_channels=config.get("in_channels", 4),
            classes=config.get("num_classes", 2),
        )
        self.lr = config.get("lr", 1e-3)
        self.loss_fn = smp.losses.DiceLoss(mode="multiclass")
        self.iou_metric = smp.utils.metrics.IoU(threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        iou = self.iou_metric(preds, y)
        self.log("train_loss", loss)
        self.log("train_iou", iou, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        iou = self.iou_metric(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_unet_lightning(config):
    return UNetLightning(config)
