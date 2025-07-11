import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import MLFlowLogger

from utils.logger import get_logger
from utils.callbacks import log_test_predictions
from model_factory import get_model

import argparse


class SnuplassPredictDataset(Dataset):
    def __init__(self, image_dir, dom_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.dom_dir = Path(dom_dir)
        self.filenames = sorted([f.name for f in self.image_dir.iterdir() if f.suffix == ".png"])
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        image = Image.open(self.image_dir / fname).convert("RGB")
        dom = Image.open(self.dom_dir / fname).convert("L")

        image = self.transform(image)  # (3, H, W)
        dom = self.transform(dom)     # (1, H, W)
        combined = torch.cat([image, dom], dim=0)  # (4, H, W)

        return combined, fname 


def main(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_names"][0]
    model_cfg = config["model"][model_name]
    data_cfg = config["data"]
    predict_cfg = config["predict"]
    log_cfg = config["logging"]

    model = get_model(model_name, model_cfg, checkpoint_path=model_cfg["checkpoint_path"])
    print("✅ Modell er lastet fra get_model()")
    model.eval()

    dataset = SnuplassPredictDataset(
        image_dir=data_cfg["image_dir"],
        dom_dir=data_cfg["dom_dir"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        shuffle=False
    )

    logger = get_logger(model_name, config)

    log_test_predictions(
        model=model,
        dataloader=dataloader,
        logger=logger,
        artifact_dir=predict_cfg["output_dir"],
        threshold=predict_cfg.get("threshold", 0.5),
        max_logs=predict_cfg.get("max_logs", 20)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str,
        required=True, 
        help="Path til YAML-konfigurasjon"
        )
    args = parser.parse_args()

    main(args.config)

