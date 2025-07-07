from lightning.pytorch.loggers import MLFlowLogger  # MLFlowLogger er ennå ikke i lightning 
import os


def get_logger(model_name: str, config: dict) -> MLFlowLogger:
    logging_config = config.get("logging", {})
    experiment_name = logging_config.get("experiment_name", "default_experiment")


    return MLFlowLogger(
        experiment_name=experiment_name,
        tags={"model": model_name}
    )