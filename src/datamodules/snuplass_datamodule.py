import os
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from dataProcessing.dataset import SnuplassDataset, load_numpy_split_stack
from dataProcessing.transform import get_train_transforms, get_val_transforms


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

        # Augmentering konfigurasjon
        use_aug = data_config.get("use_augmentation", False)
        aug_ratio = data_config.get("augmentation_ratio", None)

        self.train_transform = get_train_transforms(cfg=data_config, ratio=aug_ratio) if use_aug else None
        self.val_transform = get_val_transforms()

        if self.train_transform is not None:
            print(f"Augmentation aktivert: {self.train_transform}")
        else:
            print("Augmentation deaktivert")

        for d in [self.image_dir, self.mask_dir, self.dom_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Data-mappe finnes ikke: {d}")

    def setup(self, stage=None):
        train_ids, val_ids, _ = load_numpy_split_stack(
            self.image_dir, self.mask_dir, self.dom_dir,
            holdout_size=self.holdout_size,
            test_size=self.val_split,
            seed=self.seed,
        )

        self.train_dataset = SnuplassDataset(
            self.image_dir, self.mask_dir, self.dom_dir,
            train_ids, transform=self.train_transform,
        )

        self.val_dataset = SnuplassDataset(
            self.image_dir, self.mask_dir, self.dom_dir,
            val_ids, transform=self.val_transform,
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


def get_datamodule(data_config):
    return SnuplassDataModule(data_config)