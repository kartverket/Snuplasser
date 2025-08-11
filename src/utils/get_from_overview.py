from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
import random
import numpy as np


def get_file_list_from_overview(
    spark: SparkSession,
    catalog: str,
    schema: str,
    overview_table: str,
    id_field: str,
    require_mask: bool = True
) -> list[str]:
    """
    Henter liste av ID-er (row_hash eller nodeid) fra en ferdiglaget overview-tabell:
      - alltid image_status=DOWNLOADED AND dom_status=DOWNLOADED
      - hvis require_mask=True, også mask_status=GENERATED
    """
    qualified = f"`{catalog}`.{schema}.{overview_table}"
    df = spark.table(qualified)

    filt = (
        (F.col("image_status") == "DOWNLOADED") &
        (F.col("dom_status")   == "DOWNLOADED")
    )
    if require_mask:
        filt &= (F.col("mask_status") == "GENERATED")
    
    cols = [id_field, "image_path", "dom_path"]
    if require_mask:
        cols.append("mask_path")

    picked = df.filter(filt).select(*cols).distinct()
    
    if require_mask:
        return picked.rdd.map(lambda r: (r[id_field], r.image_path, r.dom_path, r.mask_path)).collect()
    else:
        return picked.rdd.map(lambda r: (r[id_field], r.image_path, r.dom_path)).collect()

def get_split_from_overview(
    spark: SparkSession,
    catalog: str,
    schema: str,
    overview_table: str,
    id_field: str,
    val_size: float = 0.2,
    holdout_size: int = 50,
    require_mask: bool = True,
    seed: int = 42
) -> tuple[
    list[tuple[str, str, str, str]],
    list[tuple[str, str, str, str]],
    list[tuple[str, str, str, str]]
]:
    """
    Henter alle (row_hash, image_path, dom_path, mask_path), så splitt i:
      - holdout  (første N rader)
      - train/val (resten, delt med sklearn.train_test_split)
    """
    all_items = get_file_list_from_overview(
            spark, catalog, schema, overview_table, id_field, require_mask
        )
    
    if len(all_items) < holdout_size + 2:
        raise ValueError(
            f"For få elementer ({len(all_items)}) for holdout={holdout_size} + validering."
        )
    
    random.seed(seed)
    random.shuffle(all_items)

    holdout = all_items[:holdout_size]
    remaining = all_items[holdout_size:]

    train, val = train_test_split(remaining, test_size=val_size, random_state=seed)
    
    return train, val, holdout