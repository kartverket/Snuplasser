import argparse
import torch
import os
import yaml
import mlflow
from lightning.pytorch import Trainer
from model_factory import get_model
from utils.logger import get_logger
from utils.callbacks import get_early_stopping, get_model_checkpoint, LogPredictionsCallback
from datamodules.snuplass_datamodule import get_datamodule
from utils.checkpointing import save_best_checkpoint
from mlflow.models.signature import infer_signature




def run_experiment(model_name, config):
    print(f"Trener modell: {model_name}")

    # Forbered data
    datamodule = get_datamodule(config['data'])

    # Forbered modell
    model_config = config['model'].get(model_name, {})
    model = get_model(model_name, model_config)


    # Logger
    logger = get_logger(model_name, config)

    # Callbacks
    log_pred_cfg = config.get("log_predictions_callback", {})
    log_predictions = LogPredictionsCallback(**log_pred_cfg)

    early_stopping = get_early_stopping(config['training'])
    model_checkpoint = get_model_checkpoint(config['training'])

    # Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training'].get('accelerator', 'cpu'),
        devices=config['training'].get('devices', 1),
        precision=config['training'].get('precision', 16),
        callbacks=[model_checkpoint, early_stopping, log_predictions],
        log_every_n_steps=10,
        deterministic=True  # Reproduserbarhet
    )

    # Trening og validering
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)

    #Laster den beste checkpoint etter trening
    ckpt_path= save_best_checkpoint(model_checkpoint, model_name)
    #print(f"modelname: {model_name}")

    # Laster modellen fra checkpoint for å kunne validere og logge den
    mlflow.set_registry_uri("databricks")
    with mlflow.start_run(run_id=trainer.logger.run_id):

        trained_model= model.__class__.load_from_checkpoint(
            str(ckpt_path), 
            config= model_config)
        
        trainer.validate(trained_model, datamodule=datamodule)
    # Logger valideringsmetrikker til MLflow
        val_metrics= trainer.callback_metrics
        mlflow.log_metrics({
            "val_acc": val_metrics["val_acc"].item(),
            "val_dice": val_metrics["val_dice"].item(),
            "val_iou": val_metrics["val_iou"].item(),
            "val_loss": val_metrics["val_loss"].item()

        })
    # Lagrer beste checkpoint som en artefakt
        mlflow.log_artifact(str(ckpt_path), artifact_path="best_checkpoint")
        
    # Logger hele den trente modellen til MLflow
        mlflow.pytorch.log_model(
            pytorch_model=trained_model,
            artifact_path="model",
            registered_model_name=model_name
        )


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
        help="Path til YAML-konfigurasjon"
        )
    args = parser.parse_args()
    if args.config is None:
        raise ValueError("Du må angi path til en YAML-konfigurasjon med --config")
    main(args.config)

