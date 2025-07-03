import argparse
import yaml
import mlflow
from lightning.pytorch import Trainer
import os
from model_factory import get_model
from utils.logger import get_logger
from utils.callbacks import get_early_stopping, get_model_checkpoint
from datamodules.snuplass_datamodule import get_datamodule

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

    # Callbacks
    early_stopping = get_early_stopping(config['training'])
    model_checkpoint = get_model_checkpoint(config['training'])
    # Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training'].get('accelerator', 'gpu'),
        devices=config['training'].get('devices', 1),
        precision=config['training'].get('precision', 16),
        callbacks=[model_checkpoint, early_stopping],
        log_every_n_steps=10,
        deterministic=True  # Reproduserbarhet for sammenligning av modeller
    )
    # Trening og validering
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)

    print("MLflow artifacts:", os.listdir("./src/mlruns/0"))

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        print("Konfig-innhold:", config.keys())
        

    models_to_run = config.get('model_names', [])
    for model_name in models_to_run:
        run_experiment(model_name, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="/Workspace/Users/fabian.heflo@kartverket.no/Snuplasser/src/static.yaml", 
        required=True, 
        help="Path til YAML-konfigurasjon"
        )
    args = parser.parse_args()
    main(args.config)