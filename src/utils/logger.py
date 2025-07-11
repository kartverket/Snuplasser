
from lightning.pytorch.loggers import MLFlowLogger  # MLFlowLogger er ennå ikke i lightning 
import mlflow


def get_logger(model_name: str, config: dict) -> MLFlowLogger:
    logging_config = config.get("logging", {})
    experiment_name = logging_config.get("experiment_name", "default_experiment")
    tracking_uri = logging_config.get("tracking_uri", "mlruns")  # fallback

    mlflow.set_tracking_uri(tracking_uri)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)

    return MLFlowLogger(
        experiment_name=experiment_name,
        tags={"model": model_name}
    )
