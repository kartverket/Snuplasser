from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql import DataFrame, SparkSession
from delta.tables import DeltaTable

from src.data.geometry_utils import to_wkt_2d
from src.data.log_utils import log_processed_gdb

try:
    spark = SparkSession.getActiveSession()
except:
    spark = SparkSession.builder.getOrCreate()


def write_delta_table(
    sdf: DataFrame, table: str, mode: str = "merge", id_col: str = "row_hash"
) -> None:
    """
    Skriver data til deltatabellen og opdaterer dersom data allerede finnes.
    """
    if mode == "overwrite":
        sdf.write.format("delta").option("mergeSchema", "true").mode(
            "overwrite"
        ).saveAsTable(table)
    else:
        delta_tbl = DeltaTable.forName(spark, table)

        delta_tbl.alias("target").merge(
            sdf.alias("source"), condition=f"target.{id_col} = source.{id_col}"
        ).whenMatchedUpdate(
            condition="target.ingest_time < source.ingest_time",
            set={col: f"source.{col}" for col in sdf.columns},
        ).whenNotMatchedInsert(
            values={col: f"source.{col}" for col in sdf.columns}
        ).execute()


def write_to_delta_table(
    sdf: DataFrame, gdb_name: str, table: str, log_table: str, id_col: str
):
    """
    Skriver logg med antall insert, update og deleter i deltatabellen og lagrer denne.
    """
    table_exists = False
    if spark.catalog.tableExists(table):
        delta_tbl = DeltaTable.forName(spark, table)
        version_before = delta_tbl.history(1).select("version").collect()[0][0]
        table_exists = True

    write_delta_table(sdf, table, id_col)

    if table_exists:
        version_after = delta_tbl.history(1).select("version").collect()[0][0]
        if version_after > version_before:
            metrics = delta_tbl.history(1).select("operationMetrics").collect()[0][0]
            updated = int(metrics.get("numTargetRowsUpdated", 0))
            inserted = int(metrics.get("numTargetRowsInserted", 0))
            deleted = int(metrics.get("numTargetRowsDeleted", 0))
            print(f"Updated: {updated}, Inserted: {inserted}, Deleted: {deleted}")
        else:
            print("No new Delta version found after merge.")
    else:
        inserted, updated, deleted = sdf.count(), 0, 0
        print(f"Updated: {updated}, Inserted: {inserted}, Deleted: {deleted}")

    log_processed_gdb(
        [(gdb_name, datetime.now(), inserted, updated, deleted)], log_table
    )


def write_to_sdf(
    gdb_path: str, gdb_name: str, layer: str, table: str, layer_crs: int
) -> DataFrame:
    """
    Returnerer en spark dataframe med data fra deltatabellen.
    """
    import geopandas as gpd

    gdf = (
        gpd.read_file(gdb_path, layer=layer)
        .set_crs(f"EPSG:{layer_crs}")
        .to_crs("EPSG:25833")
    )
    gdf["wkt_geometry"] = gdf["geometry"].apply(to_wkt_2d)
    gdf = gdf.drop(columns=["geometry"])

    sdf = spark.createDataFrame(gdf)
    sdf = sdf.withColumnRenamed("wkt_geometry", "geometry")
    sdf = (
        sdf.withColumn("ingest_time", current_timestamp())
        .withColumn("source_file", lit(gdb_name))
        .withColumn("source_layer", lit(layer))
        .withColumn("row_hash", sha2(concat_ws("||", *sdf.columns), 256))
    )

    target_cols = DeltaTable.forName(spark, table).toDF().columns
    sdf = sdf.select([c for c in sdf.columns if c in target_cols])

    return sdf
