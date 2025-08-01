from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from sklearn.model_selection import train_test_split
import numpy as np


def get_silver_path(spark: SparkSession, silver_table: str) -> str:
    """
    Henter filstien (locationUri) til Delta‐tabellen silver_table
    uten å involvere Sedona‐UDT-er.
    """
    tbl = spark.catalog.getTable(silver_table)
    return tbl.locationUri


def _read_silver_raw(
    spark: SparkSession,
    silver_path: str,
) -> DataFrame:
    """
    Leser rå Delta‐filer med eksplisitt snevert schema,
    slik at vi unngår å laste inn geometri‐UDT-er.
    """
    schema = StructType([
        StructField("image_status", StringType(), True),
        StructField("dom_status",   StringType(), True),
        StructField("mask_status",  StringType(), True),
        StructField("row_hash",     StringType(), True),
        StructField("nodeid",       StringType(), True),
    ])

    return (
        spark.read
             .format("delta")
             .option("mergeSchema", "false")
             .schema(schema)
             .load(silver_path)
    )


def get_file_list_from_silver(
    spark: SparkSession,
    silver_table: str,
    mode: str = "train",
) -> list[str]:
    """
    Returnerer ID-er (row_hash eller nodeid) med riktig status:
      - alltid image_status=DOWNLOADED AND dom_status=DOWNLOADED
      - i 'train'-modus i tillegg mask_status=GENERATED
    """
    path = get_silver_path(spark, silver_table)
    df   = _read_silver_raw(spark, path)

    base_filter = (
        (F.col("image_status") == "DOWNLOADED") &
        (F.col("dom_status")   == "DOWNLOADED")
    )

    if mode == "train":
        base_filter = base_filter & (F.col("mask_status") == "GENERATED")
        id_col = "row_hash"
    elif mode == "inference":
        id_col = "nodeid"
    else:
        raise ValueError("Mode must be 'train' or 'inference'")

    return (
        df.filter(base_filter)
          .select(id_col)
          .distinct()
          .rdd.flatMap(lambda r: r)  # -> List[str]
          .collect()
    )


def get_split_from_silver(
    spark: SparkSession,
    silver_table: str,
    val_size: float = 0.2,
    holdout_size: int = 5,
    seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """
    Henter alle row_hash for train-modus (krever alle status OK),
    og splitter dem i holdout, train, val.
    """
    # test at vi er i train-modus; rebruk filteret fra get_file_list
    all_ids = get_file_list_from_silver(spark, silver_table, mode="train")

    if len(all_ids) < holdout_size + 2:
        raise ValueError("For få bilder til å gjennomføre split med holdout og validering.")

    np.random.seed(seed)
    np.random.shuffle(all_ids)

    holdout_ids = all_ids[:holdout_size]
    remaining   = all_ids[holdout_size:]

    train_ids, val_ids = train_test_split(
        remaining,
        test_size=val_size,
        random_state=seed
    )

    return train_ids, val_ids, holdout_ids

