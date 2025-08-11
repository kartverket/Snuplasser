import os
import random
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split

from dataProcessing.dataset import SnuplassDataset
#from dataProcessing.transform import get_train_transforms, get_val_transforms
from utils.get_from_overview import (
    get_file_list_from_overview,
    get_split_from_overview,
)


class SnuplassDataModule(LightningDataModule):
    def __init__(
        self,
        data_config: dict,
        train_list: Optional[List[Tuple]] = None,
        val_list: Optional[List[Tuple]] = None,
        holdout_list: Optional[List[Tuple]] = None,
    ):
        super().__init__()
        # generelle innstillinger
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config.get("num_workers", 4)
        self.seed = data_config.get("seed", 42)

        # modus: 'train' eller 'predict'
        self.mode = data_config.get("mode", "train")

        # transformasjoner
        # use_aug = data_config.get("use_augmentation", False)
        # aug_ratio = data_config.get("augmentation_ratio", None)
        # if use_aug:
        #     self.train_transform = get_train_transforms(
        #         cfg=data_config, ratio=aug_ratio
        #     )
        # else:
        #     self.train_transform = get_train_transforms(cfg=data_config, ratio=None)
        # self.val_transform = get_val_transforms(cfg=data_config)
        self.train_transform = None
        self.val_transform = None

        # dataset
        self.train_list = train_list or []
        self.val_list = val_list or []
        self.holdout_list = holdout_list or []
        print(f"train_list: {len(self.train_list)}")
        print(f"val_list: {len(self.val_list)}")
        print(f"holdout_list: {len(self.holdout_list)}")
        print(self.train_list[:5])
        print(self.val_list[:5])
        print(self.holdout_list[:5])

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

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def get_datamodule(
    data_config: dict
):
    train_list = data_config.get("train_ids", [])
    val_list = data_config.get("val_ids", [])
    holdout_list = data_config.get("holdout_ids", [])
    return SnuplassDataModule(
        data_config=data_config,
        train_list=train_list,
        val_list=val_list,
        holdout_list=holdout_list
    )