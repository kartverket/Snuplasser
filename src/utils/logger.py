from lightning.pytorch.loggers import MLFlowLogger  # MLFlowLogger er ennå ikke i lightning 
import os


def get_logger(model_name: str, config: dict) -> MLFlowLogger:
    logging_config = config.get("logging", {})
    experiment_name = logging_config.get("experiment_name", "default_experiment")
    tracking_uri =  logging_config.get("tracking_uri", f"file://{os.getcwd()}/mlruns")

    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        tags={"model": model_name}
    )