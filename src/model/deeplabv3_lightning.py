import torch
from lightning.pytorch import LightningModule
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Lightning(LightningModule):
    def __init__(self,config):
        super().__init__()

        self.model = smp.DeepLabV3(
            encoder_name=config.get("backbone", "resnet50"),
            encoder_weights=None,
            in_channels=config.get("in_channels", 4),
            classes=1,
        )

        self.lr= config.get("lr", 1e-3)
        self.loss_fn=nn.BCEWithLogitsLoss()
       


    def forward(self, x):
        print("➡️ Forward pass called")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        print(f"🟢 Training step {batch_idx}")
        x, y = batch
        y = y.float()  
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        print(f"🔵 Validation step {batch_idx}")
        x, y = batch
        y = y.float().unsqueeze(1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    



def get_deeplabv3_lightning(config):
        return DeepLabV3Lightning(config=config)