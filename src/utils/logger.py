import mlflow
from datetime import datetime
from lightning.pytorch.loggers import MLFlowLogger  # MLFlowLogger er ennå ikke i lightning 
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# Hjelpefunksjon som genererer et meningsfullt navn for hver kjøring
def generate_run_name(model_name:str, config:dict)-> str:
    model_cfg=config.get("model", {}).get(model_name, {})

    backbone=model_cfg.get("backbone", "unkbackone")
    learning_rate=model_cfg.get("lr", "unklr")
    batch_size=model_cfg.get("batch_size", "unkbs")
    max_epochs=model_cfg.get("max_epochs", "unkep")

    time_str=datetime.now().strftime("%Y%m%d-%H%M")

    return f"{model_name}-{backbone}-lr{learning_rate}-bs{batch_size}-ep{max_epochs}-{time_str}"


def get_logger(model_name: str, config: dict) -> MLFlowLogger:
    username = spark.sql("SELECT current_user()").collect()[0][0]
    experiment_name = f"/Users/{username}/{model_name}"
    run_name = generate_run_name(model_name, config)
    tracking_uri = config.get("logging", {}).get("tracking_uri", "databricks")
    mlflow.set_tracking_uri(tracking_uri)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)

    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={"model": model_name}
    )
