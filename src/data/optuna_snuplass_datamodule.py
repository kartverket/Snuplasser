from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from src.data.dataset import SnuplassDataset


class SnuplassDataModule(LightningDataModule):
    def __init__(
        self,
        config: dict,
        model_name: str,
    ):
        """
        Setter opp datasett og dataloader for alle splittene.
        Argumenter:
            config (dict): konfigurasjonsfil
            model_name (str): navn på modell
        """
        super().__init__()
        data_config = config.get("data", {})
        self.batch_size = (
            config.get("model", {}).get(model_name, {}).get("batch_size", [])
        )
        self.num_workers = data_config.get("num_workers", 4)
        self.seed = data_config.get("seed", 42)
        self.mode = data_config.get("mode", "train")
        self.train_transform = None
        self.val_transform = None
        self.train_list = data_config.get("train_ids", [])
        self.val_list = data_config.get("val_ids", [])
        self.holdout_list = data_config.get("holdout_ids", [])

    def setup(self, stage):
        if self.mode == "train":
            # Fjern row_hash, behold bare paths
            train_list = [(img, dom, mask) for (_, img, dom, mask) in self.train_list]
            val_list = [(img, dom, mask) for (_, img, dom, mask) in self.val_list]
            holdout_list = [
                (img, dom, mask) for (_, img, dom, mask) in self.holdout_list
            ]

            self.train_dataset = SnuplassDataset(
                file_list=train_list, transform=self.train_transform
            )
            self.val_dataset = SnuplassDataset(
                file_list=val_list, transform=self.val_transform
            )
            self.test_dataset = SnuplassDataset(
                file_list=holdout_list, transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def get_datamodule(config: dict, model_name: str) -> LightningDataModule:
    """
    Returnerer en datamodule basert på data_config.
    Argumenter:
        data_config: konfigurasjonsfil
        model_name: navn på modell
    Returnerer:
        LightningDataModule: datamodul for dataloader
    """
    return SnuplassDataModule(config, model_name)