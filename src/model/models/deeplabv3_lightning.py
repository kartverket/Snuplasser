import torch
from lightning.pytorch import LightningModule
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
from torchmetrics.segmentation import DiceScore
from model.losses.losses import DiceBCELoss
from model.losses.loss_utils import compute_loss_weights
    

class DeepLabV3Lightning(LightningModule):
    """
    DeepLabV3 med Pytorch Lightning wrapper.
    """
    def __init__(self,config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = smp.DeepLabV3(
            encoder_name=config.get("backbone", "mobilenet_v2"),
            encoder_weights=None,
            in_channels=config.get("in_channels", 4),
            classes=1,
        )

        self.lr= config.get("lr", 1e-3)

        mask_dir = config.get("data", {}).get("mask_dir") 
        if mask_dir:
            dice_w, bce_w, pos_w = compute_loss_weights(mask_dir)
        else:
            dice_w, bce_w, pos_w = 0.7, 0.3, 1.0

        self.loss_fn = DiceBCELoss(
            dice_weight=dice_w,
            bce_weight=bce_w,
            pos_weight=pos_w,
        )

        self.iou_metric = BinaryJaccardIndex()
        self.dice = DiceScore(num_classes=2)
        self.accuracy = BinaryAccuracy()       


    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.model(x)
    
    def training_step(self, batch, _):
        x, y, _ = batch
        y = y.float()  
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y, _ = batch
        y = y.float()
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _mask, filename = batch
        with torch.no_grad():
            logits = self(x)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()
        return {"filename": filename, "mask": preds, "image": x}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
def get_deeplabv3_lightning(config):
    """
    Returnerer en instans av DeepLabV3-modellen.
    """
    return DeepLabV3Lightning(config=config)