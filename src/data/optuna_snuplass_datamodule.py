from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from src.data.dataset import SnuplassDataset


class SnuplassDataModule(LightningDataModule):
    def __init__(
        self,
        config: dict,
        model_name: str,
        train_list: Optional[List[Tuple]] = None,
        val_list: Optional[List[Tuple]] = None,
        holdout_list: Optional[List[Tuple]] = None,
    ):
        super().__init__()
        # generelle innstillinger
        data_config = config.get("data", {})
        self.batch_size = (
            config.get("model", {}).get(model_name, {}).get("batch_size", [])
        )
        self.num_workers = data_config.get("num_workers", 4)
        self.seed = data_config.get("seed", 42)

        # modus: 'train' eller 'predict'
        self.mode = data_config.get("mode", "train")

        self.train_transform = None
        self.val_transform = None

        # dataset
        self.train_list = train_list or []
        self.val_list = val_list or []
        self.holdout_list = holdout_list or []

    def setup(self, stage: Optional[str] = None):
        # TRAIN modus: split med holdout + val
        if self.mode == "train":
            # Fjern row_hash, behold bare paths
            train_list = [(img, dom, mask) for (_, img, dom, mask) in self.train_list]
            val_list = [(img, dom, mask) for (_, img, dom, mask) in self.val_list]
            holdout_list = [
                (img, dom, mask) for (_, img, dom, mask) in self.holdout_list
            ]

            # dataset-instansiering
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


def get_datamodule(data_config: dict):
    train_list = data_config.get("train_ids", [])
    val_list = data_config.get("val_ids", [])
    holdout_list = data_config.get("holdout_ids", [])
    return SnuplassDataModule(
        data_config=data_config,
        train_list=train_list,
        val_list=val_list,
        holdout_list=holdout_list,
    )