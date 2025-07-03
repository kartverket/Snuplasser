import torch
import pytorch_lightning as pl
import segmentattion_models_pytorch as smp



class DeepLabV3Lightning(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = smp.DeepLabV3(
            encoder_name=self.get("encoder", "resnet50"),
            encoder_weights=self.get("encoder_weights", "imagenet"),
            in_channels=self.get("in_channels", 4),
            classes=self.get("num_classes", 2),
        )

        self.loss_fn=smp.losses.DiceLoss(mode='multiclass')
        self.iou_metric = MulticlassJaccardIndex( num_classes=config.get("num_classes", 2))
        self.lr= config.get("lr", 1e-3)


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss= self.loss_fn(logits, y)
        preds = torch.softmax(logits, dim=1)
        iou = self.iou_metric(preds, y)
        self.log("train_loss", loss)
        self.log("train_iou", iou, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss= self.loss_fn(logits, y)
        preds = torch.softmax(logits, dim=1)
        iou = self.iou_metric(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def get_deeplabv3_lightning(config):
        """
        Factory method to create a DeepLabV3Lightning model with the given configuration.
        """
        return DeepLabV3Lightning(config)
     

    
