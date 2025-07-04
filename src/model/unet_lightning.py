import torch
from torch import nn
from lightning.pytorch import LightningModule
import segmentation_models_pytorch as smp

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
        self.loss_fn = nn.BCEWithLogitsLoss()  # passer til én output-kanal og float-target i [0, 1]

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)  # nødvendig for BCEWithLogits
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_unet_lightning(config):
    return UNetLightning(config)
