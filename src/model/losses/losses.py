import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """
    Kombinert Dice- og BCE-loss med vektede komponenter.
    Argumenter:
        dice_weight (float): Vekt for Dice loss.
        bce_weight (float): Vekt for BCE loss.
        pos_weight (float): Positiv klassevekt til BCEWithLogitsLoss.
    """

    def __init__(
        self, dice_weight: float = 0.5, bce_weight: float = 0.5, pos_weight: float = 1.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self._dice_loss(preds, targets)
        bce = self.bce(preds, targets)
        return self.dice_weight * dice + self.bce_weight * bce

    @staticmethod
    def _dice_loss(
        preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Beregner Dice loss mellom prediksjoner og mål.
        Argumenter:
            preds (Tensor): Logits fra modellen.
            targets (Tensor): Ground truth binærmaske.
        Returnerer:
            Tensor: Dice loss-verdi.
        """
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2.0 * intersection + epsilon) / (
            preds.sum() + targets.sum() + epsilon
        )