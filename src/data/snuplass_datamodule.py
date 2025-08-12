
import os
import random
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

from data.dataset import SnuplassDataset
from utils.transform import get_train_transforms, get_val_transforms
from utils.get_from_overview import (
    get_file_list_from_overview,
    get_split_from_overview,
)

class SnuplassDataModule(LightningDataModule):
    def __init__(self, data_config: dict):
        super().__init__()
        # generelle innstillinger
        self.batch_size   = data_config["batch_size"]
        self.num_workers  = data_config.get("num_workers", 4)
        self.val_split    = data_config.get("val_split", 0.2)
        self.holdout_size = data_config.get("holdout_size", 50)
        self.seed         = data_config.get("seed", 42)

        # modus: 'train' eller 'predict'
        self.mode = data_config.get("mode", "train")
        section = data_config.get(self.mode, {})
        self.overview_table = section["overview_table"]
        self.id_field       = section["id_field"]
        self.require_mask   = (self.mode == "train")

        # Spark for oversiktstabell
        self.spark = (
            SparkSession.builder
            .appName("snuplass")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )
        # sett katalog og schema om nødvendig
        self.catalog = data_config.get("spark_catalog")
        self.schema  = data_config.get("spark_schema")

        # transformasjoner
        use_aug  = data_config.get("use_augmentation", False)
        aug_ratio = data_config.get("augmentation_ratio", None)
        if use_aug:
            self.train_transform = get_train_transforms(cfg=data_config, ratio=aug_ratio)
        else:
            self.train_transform = get_train_transforms(cfg=data_config, ratio=None)
        self.val_transform = get_val_transforms(cfg=data_config)

    def setup(self, stage: Optional[str] = None):
        # TRAIN modus: split med holdout + val
        if self.mode == "train":
            # hent tuples: (row_hash, image_path, dom_path, mask_path)
            train_items, val_items, holdout_items = get_split_from_overview(
                spark          = self.spark,
                catalog        = self.catalog,
                schema         = self.schema,
                overview_table = self.overview_table,
                id_field       = self.id_field,
                val_size       = self.val_split,
                holdout_size   = self.holdout_size,
                require_mask   = True,
                seed           = self.seed
            )
            # Fjern row_hash, behold bare paths
            train_list   = [(img, dom, mask) for (_, img, dom, mask) in train_items]
            val_list     = [(img, dom, mask) for (_, img, dom, mask) in val_items]
            holdout_list = [(img, dom, mask) for (_, img, dom, mask) in holdout_items]

            # dataset-instansiering
            self.train_dataset = SnuplassDataset(
                file_list = train_list,
                transform = self.train_transform
            )
            self.val_dataset = SnuplassDataset(
                file_list = val_list,
                transform = self.val_transform
            )
            self.test_dataset = SnuplassDataset(
                file_list = holdout_list,
                transform = self.val_transform
            )

        # PREDICT modus: bruk alle rader (uten mask)
        elif self.mode == "predict":
            # hent tuples (row_hash, image_path, dom_path, mask_path)
            items = get_file_list_from_overview(
                spark          = self.spark,
                catalog        = self.catalog,
                schema         = self.schema,
                overview_table = self.overview_table,
                id_field       = self.id_field,
                require_mask   = False
            )
           
            predict_list = [(image_path, dom_path) for (_, image_path, dom_path) in items]

            self.predict_dataset = SnuplassDataset(
                file_list = predict_list,
                transform = self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


def get_datamodule(data_config: dict) -> LightningDataModule:
    return SnuplassDataModule(data_config)
