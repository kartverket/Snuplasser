import os
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from dataProcessing.dataset import SnuplassDataset
from dataProcessing.transform import get_train_transforms, get_val_transforms
from utils.get_from_silver import get_file_list_from_silver, get_split_from_silver


class SnuplassDataModule(LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.image_dir = data_config["image_dir"]
        self.mask_dir = data_config["mask_dir"]
        self.dom_dir = data_config["dom_dir"]
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config.get("num_workers", 4)
        self.val_split = data_config.get("val_split", 0.2)
        self.holdout_size = data_config.get("holdout_size", 5)
        self.seed = data_config.get("seed", 42)
        self.train_transform = data_config.get("train_transform", None)
        self.val_transform = data_config.get("val_transform", None)
        self.silver_table = data_config["silver_table"]
        self.mode = data_config.get("mode", "train")


        # Augmentering konfigurasjon
        use_aug = data_config.get("use_augmentation", False)
        aug_ratio = data_config.get("augmentation_ratio", None)

        self.train_transform = get_train_transforms(cfg=data_config, ratio=aug_ratio) if use_aug else None
        self.val_transform = get_val_transforms()

        if self.train_transform is not None:
            print(f"Augmentation aktivert: {self.train_transform}")
        else:
            print("Augmentation deaktivert")

        for d in [self.image_dir, self.dom_dir] + ([self.mask_dir] if self.mode == "train" and self.mask_dir else []):
            if d and not os.path.isdir(d):
                raise FileNotFoundError(f"Data-mappe finnes ikke: {d}")

    def setup(self, stage=None):
        all_ids = get_file_list_from_silver(silver_table=self.silver_table, mode=self.mode)

        if self.mode == "train":
            train_ids, val_ids, holdout_ids = get_split_from_silver(
                silver_table=self.silver_table,
                val_size=self.val_split,
                holdout_size=self.holdout_size,
                seed=self.seed
            )

            self.train_dataset = SnuplassDataset(
                self.image_dir, self.mask_dir, self.dom_dir,
                train_ids, transform=self.train_transform,
            )

            self.val_dataset = SnuplassDataset(
                self.image_dir, self.mask_dir, self.dom_dir,
                val_ids, transform=self.val_transform,
            )
        elif self.mode == "inference":
            self.inference_dataset = SnuplassDataset(
                image_dir=self.image_dir,
                mask_dir=None,
                dom_dir=self.dom_dir,
                file_list=all_ids,
                transform=self.val_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def inference_dataloader(self):
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


def get_datamodule(data_config):
    return SnuplassDataModule(data_config)