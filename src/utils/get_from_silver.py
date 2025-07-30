from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split


def get_file_list_from_silver(silver_table: str, mode: str = "train") -> list[str]:
    """
    Returnerer en liste med ID-er (row_hash eller nodeid) der bildene er klare.
    Krever ikke mask_status for inference-mode.
    """
    spark = SparkSession.getActiveSession()
    if not spark:
        raise RuntimeError("SparkSession is not active.")

    df = spark.read.table(silver_table)
    columns = df.columns

    # Filter avhenger av modus
    if mode == "train":
        # Krever at alle 3 typer data er klare
        required_filter = (
            "image_status = 'DOWNLOADED' AND dom_status = 'DOWNLOADED' AND mask_status = 'GENERATED'"
        )
        id_col = "row_hash"
    elif mode == "inference":
        # Bare DOM og ortofoto
        required_filter = (
            "image_status = 'DOWNLOADED' AND dom_status = 'DOWNLOADED'"
        )
        id_col = "nodeid"
    else:
        raise ValueError("Mode must be 'train' or 'inference'")

    # Hvis mask_status ikke finnes og vi er i train-modus → feil
    if mode == "train" and "mask_status" not in columns:
        raise ValueError("mask_status mangler i tabellen – kan ikke brukes til trening")

    # Filtrer og hent id-kolonne
    df = df.filter(required_filter)
    return [row[id_col] for row in df.select(id_col).distinct().collect()]


def get_split_from_silver(silver_table: str, val_size=0.2, holdout_size=5, seed=42):
    spark = SparkSession.getActiveSession()
    if not spark:
        raise RuntimeError("SparkSession is not active.")

    df = spark.read.table(silver_table).filter(
        "image_status = 'DOWNLOADED' AND dom_status = 'DOWNLOADED' AND mask_status = 'GENERATED'"
    )

    row_hashes = [row["row_hash"] for row in df.select("row_hash").distinct().collect()]
    if len(row_hashes) < holdout_size + 2:
        raise ValueError("For få bilder til å gjennomføre splitting med holdout og validering.")

    import numpy as np
    np.random.seed(seed)
    np.random.shuffle(row_hashes)

    holdout_ids = row_hashes[:holdout_size]
    remaining = row_hashes[holdout_size:]
    train_ids, val_ids = train_test_split(remaining, test_size=val_size, random_state=seed)

    return train_ids, val_ids, holdout_ids
