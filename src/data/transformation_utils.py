from pyproj import CRS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *

try:
    spark = SparkSession.getActiveSession()
except:
    spark = SparkSession.builder.getOrCreate()


def get_srid(catalog: str, schema: str, table: str):
    """
    Henter ut SRID fra metadata tags.
    """
    query = f"""
        SELECT tag_value
        FROM system.information_schema.table_tags
        WHERE catalog_name = '{catalog}'
          AND schema_name = '{schema}'
          AND table_name = '{table}'
          AND tag_name = 'SRID'
    """
    result = spark.sql(query).collect()
    return result[0]["tag_value"] if result else None


def crs_is_righthanded(srid: str) -> bool:
    """
    Sjekker om koordinatsystemet er høyrehåndsystem.
    Dvs. rekkefølge øst, nord
    """
    return CRS(srid).axis_info[0].direction.upper() == "EAST"


def transform_to_epsg(
    df: DataFrame,
    col: str = "geometry",
    source_srid: str = "EPSG:5942",
    target_srid: str = "EPSG:25833",
) -> DataFrame:
    """
    Transformerer geometri til ønsket EPSG-projeksjon.
    """
    # Flipper koordinatene x og y hvis koordinatsystemet er venstrehåndsystem.
    if crs_is_righthanded(source_srid) == False:
        df = df.withColumn(
            col,
            expr(
                f"ST_FlipCoordinates(ST_SetSRID(ST_GeomFromWKB({col}), {source_srid[5:]}))"
            ),
        )
    else:
        df = df.withColumn(
            col, expr(f"ST_SetSRID(ST_GeomFromWKB({col}), {source_srid[5:]})")
        )

    # Transformerer fra source_srid til target_srid (EPSG)
    df = df.withColumn(col, expr(f"ST_Transform({col}, '{target_srid}')"))

    # Flipper koordinatene tilbake hvis de transformerte koordinatene er venstrehåndsystem.
    if crs_is_righthanded(target_srid) == False:
        df = df.withColumn(col, expr(f"ST_FlipCoordinates({col})"))

    df = df.select(col, "kommunenummer")
    return df
