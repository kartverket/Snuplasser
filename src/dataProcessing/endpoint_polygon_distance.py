import requests
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType
import time


def fetch_endpoints_pandas(batch_iter):
    """
    batch_iter: iterator of pandas.DataFrame, each with columns "id","bbox"
    yields pandas.DataFrame with columns "id","endpoint_wkt"
    """
    for pdf in batch_iter:
        print(len(pdf))
        out_rows = []
        url = "https://nvdbapiles.atlas.vegvesen.no/vegnett/api/v4/noder"
        headers = {"Accept": "application/json", "X-Client": "Systemet for vegobjekter"}
        srid = "UTM33"

        for _, row in pdf.iterrows():
            params = {"srid": srid, "kartutsnitt": ",".join(map(str, row["bbox"]))}
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            time.sleep(3)
            resp.raise_for_status()
            for obj in resp.json().get("objekter", []):
                for port in obj.get("porter", []):
                    if port.get("tilkobling", {}).get("portnummer") in (1, 2):
                        wkt = obj["geometri"]["wkt"]
                        if not wkt.upper().startswith("POINT"):
                            wkt = wkt.split(" ", 1)[1]
                        out_rows.append(
                            {"row_hash": row["row_hash"], "endpoint_wkt": wkt}
                        )
                        break

        if out_rows:
            yield pd.DataFrame(out_rows)
        else:
            # Hvis det ikke finnes noen endepunkter, returner en tom dataframe
            yield pd.DataFrame([], columns=["row_hash", "endpoint_wkt"])


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("turnaround_distances")
        .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config(
            "spark.kryo.registrator",
            "org.apache.sedona.core.serde.SedonaKryoRegistrator",
        )
        .getOrCreate()
    )

    # Leser fra polygons silver delta tabell
    polygons_df = (
        spark.read.table("`land_topografisk-gdb_dev`.ai2025.polygons_silver")
        .select("row_hash", "geometry", "bbox")
        .withColumn("poly_wkt", F.expr("ST_AsText(geometry)"))
    )

    # Definerer schema for output
    ep_schema = StructType(
        [
            StructField("row_hash", StringType(), False),
            StructField("endpoint_wkt", StringType(), False),
        ]
    )

    # Henter endepunkter ved snuplassene
    endpoints_df = polygons_df.select("row_hash", "bbox").mapInPandas(
        fetch_endpoints_pandas, schema=ep_schema
    )

    # Regner ut avstandene
    joined = (
        polygons_df.join(endpoints_df, "row_hash", "inner")
        .withColumn("poly_geom", F.expr("ST_GeomFromWKT(poly_wkt)"))
        .withColumn("end_geom", F.expr("ST_GeomFromWKT(endpoint_wkt)"))
        .withColumn("distance_m", F.expr("ST_Distance(poly_geom, end_geom)"))
    )

    # Samler resultatene tilhørende samme polygon og finner de lengste avstandene
    result = joined.groupBy("row_hash").agg(F.min("distance_m").alias("distance_m"))
    result.orderBy(F.desc("distance_m")).show(10, truncate=False)

    spark.stop()