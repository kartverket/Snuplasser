import torch

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, ignore: int = None):
    SMOOTH = 1e-6

    # If ignore parameter is provided, mask those values
    if ignore is not None:
        outputs = torch.where(outputs == ignore, torch.zeros_like(outputs), outputs)
        labels = torch.where(labels == ignore, torch.zeros_like(labels), labels)

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our division to avoid 0/0

    return iou.mean() 



def acc_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    if outputs.dim() > 2:
        outputs = outputs.squeeze(1)

    acc = torch.sum(outputs == labels) / (labels.size(0) * labels.size(1) * labels.size(2))

    return acc

