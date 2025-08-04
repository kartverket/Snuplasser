from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
import numpy as np


def get_file_list_from_overview(
    spark: SparkSession,
    overview_table: str,
    id_field: str,
    require_mask: bool = True
) -> list[str]:
    """
    Henter liste av ID-er (row_hash eller nodeid) fra en ferdiglaget overview-tabell:
      - alltid image_status=DOWNLOADED AND dom_status=DOWNLOADED
      - hvis require_mask=True, også mask_status=GENERATED
    """
    df = spark.table(overview_table)

    base_filter = (
        (F.col("image_status") == "DOWNLOADED") &
        (F.col("dom_status")   == "DOWNLOADED")
    )
    if require_mask:
        base_filter &= (F.col("mask_status") == "GENERATED")

    return (
        df.filter(base_filter)
          .select(F.col(id_field))
          .distinct()
          .rdd.flatMap(lambda r: r)  # -> List[str]
          .collect()
    )


def get_split_from_overview(
    spark: SparkSession,
    overview_table: str,
    id_field: str,
    val_size: float = 0.2,
    holdout_size: int = 5,
    seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """
    Samme som get_file_list, men splitter i train/val/holdout.
    """
    all_ids = get_file_list_from_overview(
        spark, overview_table, id_field, require_mask=True
    )

    if len(all_ids) < holdout_size + 2:
        raise ValueError(
            "For få elementer til å lage holdout + validering."
        )

    np.random.seed(seed)
    np.random.shuffle(all_ids)

    holdout = all_ids[:holdout_size]
    remaining = all_ids[holdout_size:]

    train, val = train_test_split(
        remaining, 
        test_size=val_size, 
        random_state=seed
    )

    return train, val, holdout