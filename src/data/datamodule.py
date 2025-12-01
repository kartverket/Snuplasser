from typing import Optional
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from pyspark.sql import SparkSession

from data.dataset import SnuplassDataset, HelicopterDataset
from utils.transform import get_train_transforms, get_val_transforms
from utils.get_from_overview import (
    get_file_list_from_overview,
    get_split_from_overview,
)


class DataModule(LightningDataModule):
    def __init__(self, config: dict, model_name: str):
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
        self.val_split = data_config.get("val_split", 0.2)
        self.holdout_size = data_config.get("holdout_size", 50)
        self.seed = data_config.get("seed", 42)
        self.mode = data_config.get("mode", "train")
        section = data_config.get(self.mode, {})
        self.overview_table = section["overview_table"]
        self.id_field = section["id_field"]
        self.require_mask = self.mode == "train"
        self.target = section["target"]

        # Spark for oversiktstabell
        self.spark = (
            SparkSession.builder.appName("snuplass")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )
        self.catalog = data_config.get("spark_catalog")
        self.schema = data_config.get("spark_schema")

        # Transformasjoner
        use_aug = data_config.get("use_augmentation", False)
        aug_ratio = data_config.get("augmentation_ratio", None)
        if use_aug:
            self.train_transform = get_train_transforms(
                cfg=data_config, ratio=aug_ratio
            )
        else:
            self.train_transform = get_train_transforms(cfg=data_config, ratio=None)
        self.val_transform = get_val_transforms(cfg=data_config)

    def setup(self, stage):
        if self.mode == "train":
            # hent tuples: (row_hash, image_path, dom_path, mask_path)
            train_items, val_items, holdout_items = get_split_from_overview(
                spark=self.spark,
                catalog=self.catalog,
                schema=self.schema,
                overview_table=self.overview_table,
                id_field=self.id_field,
                val_size=self.val_split,
                holdout_size=self.holdout_size,
                require_mask=True,
                seed=self.seed,
            )

            # Fjern row_hash, behold bare paths
            if self.target == "helipads":
                train_list = [(img, mask) for (_, img, mask) in train_items]
                val_list = [(img, mask) for (_, img, mask) in val_items]
                holdout_list = [(img, mask) for (_, img, mask) in holdout_items]

                self.train_dataset = HelicopterDataset(
                    file_list=train_list, transform=self.train_transform
                )
                self.val_dataset = HelicopterDataset(
                    file_list=val_list, transform=self.val_transform
                )
                self.test_dataset = HelicopterDataset(
                    file_list=holdout_list, transform=self.val_transform
                )
            else:
                train_list = [(img, dom, mask) for (_, img, dom, mask) in train_items]
                val_list = [(img, dom, mask) for (_, img, dom, mask) in val_items]
                holdout_list = [(img, dom, mask) for (_, img, dom, mask) in holdout_items]

                self.train_dataset = SnuplassDataset(
                    file_list=train_list, transform=self.train_transform
                )
                self.val_dataset = SnuplassDataset(
                    file_list=val_list, transform=self.val_transform
                )
                self.test_dataset = SnuplassDataset(
                    file_list=holdout_list, transform=self.val_transform
                )

        elif self.mode == "predict":
            # hent tuples (row_hash, image_path, dom_path)
            items = get_file_list_from_overview(
                spark=self.spark,
                catalog=self.catalog,
                schema=self.schema,
                overview_table=self.overview_table,
                id_field=self.id_field,
                require_mask=False,
            )

            if self.target == "helipads":
                predict_list = [
                    (image_path) for (_, image_path) in items
                ]
                self.predict_dataset = HelicopterDataset(
                    file_list=predict_list, transform=self.val_transform
                )
            else:
                predict_list = [
                    (image_path, dom_path) for (_, image_path, dom_path) in items
                ]
                self.predict_dataset = SnuplassDataset(
                    file_list=predict_list, transform=self.val_transform
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


def get_datamodule(config: dict, model_name: str) -> LightningDataModule:
    """
    Returnerer en datamodule basert på data_config.
    Argumenter:
        data_config: konfigurasjonsfil
        model_name: navn på modell
    Returnerer:
        LightningDataModule: datamodul for dataloader
    """
    return DataModule(config, model_name)