
from lightning.pytorch.loggers import MLFlowLogger  # MLFlowLogger er ennå ikke i lightning 
import mlflow
import os
from datetime import datetime

# Hjelpefunksjon som genererer et meningsfullt navn for hver MLflow-run
# basert på modellnavn, treningsparametere, datasett og tidsstempel.
def generate_run_name(model_name: str, config: dict) -> str:
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    ep = training_cfg.get("max_epochs", "unk")
    baseS = data_cfg.get("batch_size", "unk")
    numwork = data_cfg.get("num_workers", "unk")

    time_str = datetime.now().strftime("%m%d")

    return f"{model_name}-{ep}epoch-{baseS}baseS-{numwork}numWork-{time_str}"


def get_logger(model_name: str, config: dict) -> MLFlowLogger:
    experiment_name = config.get("logging", {}).get("experiment_name", "default_experiment")
    run_name = generate_run_name(model_name, config)

    tracking_uri = config.get("logging", {}).get("tracking_uri", None)
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)

    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={"model": model_name}
    )
