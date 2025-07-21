
from lightning.pytorch.loggers import MLFlowLogger  # MLFlowLogger er ennå ikke i lightning 
import mlflow
import os
from datetime import datetime

# Hjelpefunksjon som genererer et meningsfullt navn for hver MLflow-run
# basert på modellnavn, treningsparametere, datasett og tidsstempel.
def generate_run_name(model_name:str, config:dict)-> str:
    training_cfg=config.get("training", {})
    data_cfg=config.get("data", {})

    lr=training_cfg.get("lr", "unklr")
    bs=training_cfg.get("batch_size", "unkbs")
    ep=training_cfg.get("max_epochs", "unkep")
    ds=data_cfg.get("name","unkdata")

    time_str=datetime.now().strftime("%Y%m%d-%H%M")

    return f"{model_name}-lr{lr}-bs{bs}-ep{ep}-ds{ds}-{time_str}"



def get_logger(model_name: str, config: dict) -> MLFlowLogger:
    experiment_name = config.get("logging", {}).get("experiment_name", "default_experiment")
    run_name = generate_run_name(model_name, config)

    mlflow.set_tracking_uri(tracking_uri)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)

    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={"model": model_name}
    )
