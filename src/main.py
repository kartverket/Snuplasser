import argparse
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from lightning.pytorch import Trainer
from utils.model_factory import get_model
from utils.logger import get_logger
from utils.callbacks import (
    get_early_stopping,
    get_model_checkpoint,
    LogPredictionsCallback,
    log_predictions_from_preds,
)
from data.snuplass_datamodule import get_datamodule
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def run_experiment(model_name, config):
    mode = config.get("data", {}).get("mode", "train")
    print(f"Kjører {mode}-jobb for modell: {model_name}")

    # --- Data & modell ---
    datamodule = get_datamodule(config.get("data", {}))
    model_cfg = config.get("model", {}).get(model_name, {})
    model = get_model(model_name, model_cfg)

    
    # --- Logger & callbacks ---
    logger = get_logger(model_name, config)
    es_cb = get_early_stopping(config.get("training", {}))
    ckpt_cb = get_model_checkpoint(config.get("training", {}))
    log_pred_cfg = config.get("log_predictions_callback", {})
    log_pred_cb = LogPredictionsCallback(**log_pred_cfg)

    trainer = Trainer(
        logger=logger,
        max_epochs=config.get("training", {}).get("max_epochs", 1),
        accelerator=config.get("training", {}).get("accelerator", "cpu"),
        devices=config.get("training", {}).get("devices", 1),
        precision=config.get("training", {}).get("precision", 32),
        callbacks=[ckpt_cb, es_cb, log_pred_cb] if mode == "train" else [],
        log_every_n_steps=config.get("training", {}).get("log_every_n_steps", 10),
        deterministic=True,
    )

    if mode == "train":
        # 1) Tren + test
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

        # 2) Last inn beste checkpoint og logg til MLflow
        best_ckpt = ckpt_cb.best_model_path
        mlflow.set_registry_uri(config.get("logging", {}).get("tracking_uri", ""))
        with mlflow.start_run(run_id=trainer.logger.run_id):
            trained = model.__class__.load_from_checkpoint(
                str(best_ckpt), config=model_cfg
            )
            # valider igjen for å få metrics
            trainer.validate(trained, datamodule=datamodule)
            metrics = trainer.callback_metrics
            mlflow.log_metrics({
                #"val_acc":  metrics["val_acc"].item(),
                "val_dice": metrics["val_dice"].item(),
                "val_iou":  metrics["val_iou"].item(),
                "val_loss": metrics["val_loss"].item(),
            })
            mlflow.log_artifact(str(best_ckpt), artifact_path="best_checkpoint")
            mlflow.pytorch.log_model(
                pytorch_model=trained,
                artifact_path="model",
                registered_model_name=model_name,
            )

    elif mode == "predict":
        # Last inn checkpoint for prediksjon
        username = spark.sql("SELECT current_user()").collect()[0][0]
        experiment_name = f"/Users/{username}/{model_name}"
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        experiment_path = Path(f"/Workspace/Users/{username}/Snuplasser/src/{experiment_id}")

        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=100
        )

        filtered_runs = []
        for run in runs:
            experiment_folders = [f for f in experiment_path.iterdir() if f.is_dir()]
            for folder in experiment_folders:
                if run.info.run_id in folder.name:
                    filtered_runs.append(run)

        if filtered_runs:
            newest = filtered_runs[0]
        else:
            raise ValueError("Fant ingen kjøringer i dette eksperimentet")

        ckpt_path = f"/Workspace/Users/{username}/Snuplasser/src/{experiment_id}/{newest.info.run_id}/checkpoints/best.ckpt"
        trained = model.__class__.load_from_checkpoint(str(ckpt_path))

        # Kjør prediksjon
        preds = trainer.predict(trained, datamodule=datamodule)
        log_predictions_from_preds(preds, logger)

    else:
        raise ValueError(f"Ukjent mode: {mode}")


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Konfig-innhold:", config.keys())

    for name in config.get("model_names", []):
        run_experiment(name, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path til YAML-konfigurasjon"
    )
    args = parser.parse_args()
    main(args.config)
