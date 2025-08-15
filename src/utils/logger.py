import mlflow
from datetime import datetime
from lightning.pytorch.loggers import MLFlowLogger
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def generate_run_name(model_name:str, config:dict)-> str:
    """
    Hjelpefunksjon som genererer et meningsfullt navn for hver kjøring.
    Argumenter:
        model_name: navnet på modellen
        config: konfigurasjonsfilen
    Returnerer:
        navnet på kjøringen
    """
    model_cfg=config.get("model", {}).get(model_name, {})
    training_cfg=config.get("training", {})

    backbone=model_cfg.get("backbone", "unkbackone")
    learning_rate=model_cfg.get("lr", "unklr")
    batch_size=model_cfg.get("batch_size", "unkbs")
    max_epochs=training_cfg.get("max_epochs", "unkep")

    time_str=datetime.now().strftime("%Y%m%d-%H%M")

    return f"{model_name}-{backbone}-lr{learning_rate}-bs{batch_size}-ep{max_epochs}-{time_str}"


def get_logger(model_name: str, config: dict) -> "MLFlowLogger":
    """
    Hjelpefunksjon som returnerer en MLFlowLogger.
    Argumenter:
        model_name: navnet på modellen
        config: konfigurasjonsfilen
    Returnerer:
        MLFlowLogger objekt med riktig experiment og run_name
    """
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
