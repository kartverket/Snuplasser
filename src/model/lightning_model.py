import lightning as L
import torch
import torch.nn.functional as F

from utils import acc_pytorch as acc
from losses import DiceLoss
from model import dcswin_tiny, dcswin_small, dcswin_base

# from newformer import NewFormer

from statistics import mean

from unet import UNet


# BEGIN: qe7d5f8g4hj2
import torch
import numpy as np
from PIL import Image


def change_values_and_save(tensor, filename, mask=False):
    # Replace values
    if mask:
        tensor[tensor == 1] = 100
        tensor[tensor == 2] = 200
        tensor[tensor == 3] = 255

    # Convert tensor to numpy array
    np_array = tensor.cpu().numpy().astype(np.uint8)

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(np_array)

    # Save PIL Image to file
    pil_image.save(filename)


# END: qe7d5f8g4hj2


class DCSwin(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model_size: str = "tiny",
        model_name: str = "unet",
    ):
        super().__init__()

        if model_name == "dcswin":
            if model_size == "tiny":
                self.model = dcswin_tiny(
                    True,
                    num_classes=num_classes,
                    weight_path=f"pretrain_weights/stseg_{model_size}.pth",
                )
            elif model_size == "small":
                self.model = dcswin_small(
                    False,
                    num_classes=num_classes,
                    weight_path=f"pretrain_weights/stseg_{model_size}.pth",
                )
            elif model_size == "base":
                self.model = dcswin_base(
                    True,
                    num_classes=num_classes,
                    weight_path=f"pretrain_weights/stseg_{model_size}.pth",
                )
            else:
                raise NotImplementedError("Model size not implemented")
        if model_name == "unet":
            self.model = UNet(3, num_classes, bilinear=False)

        # self.model = NewFormer()
        # self.model = dcswin_tiny(False, num_classes=num_classes, weight_path=f"pretrained_weights/stseg_{model_size}.pth")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.loss = DiceLoss(mode="multiclass", ignore_index=self.num_classes)

        # Training metrics
        self.train_iou = list()
        self.train_acc = list()
        self.train_loss = list()

        # Validation metrics
        self.val_iou = list()
        self.val_acc = list()
        self.val_loss = list()

    def forward(self, x):
        return self.model(x)

    def calculate_metrics(self, logits, mask, step_type="train"):
        prediction = F.softmax(logits, dim=1).argmax(dim=1)

        intersection = torch.logical_and(prediction, mask).sum(dim=(1, 2))
        union = torch.logical_or(prediction, mask).sum(dim=(1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)

        miou = iou.mean()
        macc = acc(prediction, mask)

        if step_type == "train":
            self.train_iou.append(miou.item())
            self.train_acc.append(macc.item())
        else:
            self.val_iou.append(miou.item())
            self.val_acc.append(macc.item())

    def on_train_epoch_end(self):
        if len(self.train_iou) > 0:
            epoch_iou = mean(self.train_iou)
        else:
            epoch_iou = 0
        if len(self.train_acc) > 0:
            epoch_acc = mean(self.train_acc)
        else:
            epoch_acc = 0
        if len(self.train_loss) > 0:
            epoch_loss = mean(self.train_loss)
        else:
            epoch_loss = 0

        self.log("train_loss", epoch_loss, on_epoch=True, sync_dist=True)
        self.log("train_iou", epoch_iou, on_epoch=True, sync_dist=True)
        self.log("train_acc", epoch_acc, on_epoch=True, sync_dist=True)

        print(
            f"Training stats ({self.current_epoch}) | Loss: {epoch_loss}, IoU: {epoch_iou}, Acc: {epoch_acc} \n"
        )

    def on_validation_epoch_end(self):
        if len(self.val_iou) > 0:
            epoch_iou = mean(self.val_iou)
        else:
            epoch_iou = 0
        if len(self.val_acc) > 0:
            epoch_acc = mean(self.val_acc)
        else:
            epoch_acc = 0
        if len(self.val_loss) > 0:
            epoch_loss = mean(self.val_loss)
        else:
            epoch_loss = 0

        self.log("val_loss", epoch_loss, on_epoch=True, sync_dist=True)
        self.log("val_iou", epoch_iou, on_epoch=True, sync_dist=True)
        self.log("val_acc", epoch_acc, on_epoch=True, sync_dist=True)

        print(
            f"Validation stats ({self.current_epoch}) | Loss: {epoch_loss}, IoU: {epoch_iou}, Acc: {epoch_acc} \n"
        )

    def training_step(self, batch, batch_idx):
        image, mask, _ = batch

        loss = torch.zeros(1).to(self.device)

        x = self(image)

        loss = loss + self.loss(x, mask)

        self.train_loss.append(loss.item())

        self.calculate_metrics(x, mask, step_type="train")

        return loss

    def validation_step(self, batch, batch_idx):
        image, mask, _ = batch

        x = self(image)

        loss = self.loss(x, mask)
        self.val_loss.append(loss.item())

        self.calculate_metrics(x, mask, step_type="val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        return [optimizer], [scheduler]
