import random
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split


def get_file_list_from_overview(
    spark: SparkSession,
    catalog: str,
    schema: str,
    overview_table: str,
    id_field: str,
    require_mask: bool = True
) -> list[str]:
    """
    Henter liste av ID-er (row_hash eller nodeid) fra en ferdiglaget overview-tabell.
    Argumenter:
        spark (SparkSession): SparkSession som brukes til å lese data
        catalog (str): Navnet på catalog
        schema (str): Navnet på schema
        overview_table (str): Navnet på oversiktstabellen
        id_field (str): Navnet på feltet som brukes som ID
        require_mask (bool): Om masken må være generert for alle filene
    Returnerer:
        list[str]: Liste av ID-er
    """
    qualified = f"`{catalog}`.{schema}.{overview_table}"
    df = spark.table(qualified)

    if "dom_status" in df.columns:
        filt = (
            (F.col("image_status") == "DOWNLOADED") &
            (F.col("dom_status")   == "DOWNLOADED")
        )
        cols = [id_field, "image_path", "dom_path"]
    else:
        filt = (F.col("image_status") == "DOWNLOADED")
        cols = [id_field, "image_path"]
    
    if require_mask:
        filt &= (F.col("mask_status") == "GENERATED")
        cols.append("mask_path")

    picked = df.filter(filt).select(*cols).distinct()
    
    rows = picked.collect()

    if require_mask and "dom_path" in df.columns:
        return [(r[id_field], r["image_path"], r["dom_path"], r["mask_path"]) for r in rows]
    elif "dom_path" in df.columns:
        return [(r[id_field], r["image_path"], r["dom_path"]) for r in rows]
    elif require_mask:
        return [(r[id_field], r["image_path"], r["mask_path"]) for r in rows]
    else:
        return [(r[id_field], r["image_path"]) for r in rows]

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
    Henter data fra en ferdiglaget overview-tabell og splitter den i train, val og holdout.
    Argumenter:
        spark (SparkSession): SparkSession som brukes til å lese data
        catalog (str): Navnet på catalog
        schema (str): Navnet på schema
        overview_table (str): Navnet på oversiktstabellen
        id_field (str): Navnet på feltet som brukes som ID
        val_size (float): Prosentandel av data som skal brukes til validering
        holdout_size (int): Antall elementer som skal brukes til holdout
        require_mask (bool): Om masken må være generert for alle filene
        seed (int): Seed for tilfeldige valg
    Returnerer:
        tuple: Tuple med tre lister med ID-er for train, val og holdout
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