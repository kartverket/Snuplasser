import os
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    IntegerType,
)
from pyspark.sql import SparkSession

try:
    spark = SparkSession.getActiveSession()
except:
    spark = SparkSession.builder.getOrCreate()


def log_processed_gdb(log_data: list, log_table: str):
    """
    Skriver logg med antall insert, update og deleter i deltatabellen.
    """
    schema = StructType(
        [
            StructField("gdb_name", StringType(), True),
            StructField("processed_time", TimestampType(), True),
            StructField("num_inserted", IntegerType(), True),
            StructField("num_updated", IntegerType(), True),
            StructField("num_deleted", IntegerType(), True),
        ]
    )
    spark.createDataFrame(log_data, schema=schema).write.format("delta").mode(
        "append"
    ).saveAsTable(log_table)


def log_predicted_masks(log_data: list, log_table: str):
    """
    Skriver logg med antall insert, update og deleter i deltatabellen.
    """
    schema = StructType(
        [
            StructField("processed_time", TimestampType(), True),
            StructField("num_inserted", IntegerType(), True),
            StructField("num_updated", IntegerType(), True),
            StructField("num_deleted", IntegerType(), True),
        ]
    )
    spark.createDataFrame(log_data, schema=schema).write.format("delta").mode(
        "append"
    ).saveAsTable(log_table)


def check_for_new_gdbs(landing_zone: str, log_table: str, obj: str) -> list:
    """
    Returnerer en liste med geodatabaser som ikke er lagret i deltatabellen.
    """
    all_gdbs = [
        os.path.join(landing_zone, f) for f in os.listdir(landing_zone) if obj in f
    ]
    processed_gdbs_df = spark.read.table(log_table).select("gdb_name")
    processed_gdbs = [row["gdb_name"] for row in processed_gdbs_df.collect()]

    return [gdb for gdb in all_gdbs if gdb not in processed_gdbs]


def check_for_new_predicted_masks(mask_path: str, table: str) -> list:
    """
    Returnerer en liste med masker som ikke er lagret i deltatabellen.
    """
    all_masks = [os.path.basename(f) for f in os.listdir(mask_path)]

    processed_masks_df = spark.read.table(table).select("source_file")
    processed_masks = [row["source_file"] for row in processed_masks_df.collect()]

    return [mask for mask in all_masks if mask not in processed_masks]
