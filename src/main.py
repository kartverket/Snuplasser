import argparse
import yaml
import mlflow
from lightning import Trainer
from model_factory import get_model
from utils.logger import get_logger

# Midlertidig datamodul-import (kan erstattes når dataoppsett er klart)
from datamodule import get_datamodule

def run_experiment(model_name, config):
    print(f"Trener modell: {model_name}")

    # Logg automatisk 
    mlflow.pytorch.autolog()

    # Forbered data (placeholder)
    datamodule = get_datamodule(config['data'])

    # Forbered modell
    model_config = config['model'].get(model_name, {})
    model = get_model(model_name, model_config)

    # Logger
    logger = get_logger(model_name, config)

    # Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=config['training']['max_epochs'],
        gpus=config['training'].get('gpus', 1),
        precision=config['training'].get('precision', 32),
        log_every_n_steps=10
    )

    # Trening og validering
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    models_to_run = config.get('model_names', [])
    for model_name in models_to_run:
        run_experiment(model_name, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="static.yaml", 
        required=True, 
        help="Path til YAML-konfigurasjon"
        )
    args = parser.parse_args()
    main(args.config)
