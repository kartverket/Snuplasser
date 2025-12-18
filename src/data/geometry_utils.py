import random
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructType,
    StructField,
)


def to_wkt_2d(geom):
    """
    Konverterer geometri til WKT format i 2D format.
    """
    from shapely.geometry.base import BaseGeometry
    from shapely import force_2d

    if isinstance(geom, BaseGeometry):
        return force_2d(geom).wkt
    return None


def random_adjusted_bbox_centered(
    envelope: list, bbox_length: int = 128, max_offset: float = 20
) -> list:
    """
    Genererer en tilfeldig justert boks rundt polygonen med en maksimal avstand fra sentrum.
    """
    xmin, ymin, xmax, ymax = envelope
    poly_width = xmax - xmin
    poly_height = ymax - ymin

    if poly_width > bbox_length or poly_height > bbox_length:
        print("OBS: polygon er større enn bbox")
        max_offset = 0

    half_size = bbox_length / 2

    center_x = (xmin + xmax) / 2 + random.uniform(-max_offset, max_offset)
    center_y = (ymin + ymax) / 2 + random.uniform(-max_offset, max_offset)

    adjusted_xmin = center_x - half_size
    adjusted_xmax = center_x + half_size
    adjusted_ymin = center_y - half_size
    adjusted_ymax = center_y + half_size

    bbox = [adjusted_xmin, adjusted_ymin, adjusted_xmax, adjusted_ymax]
    bbox_str = ",".join(f"{v:.6f}" for v in bbox)
    return bbox, bbox_str


def make_bbox(df: DataFrame, bbox_length: int) -> DataFrame:
    """
    Lager en boks rundt polygonen med en fast avstand fra sentrum.
    """
    df = df.withColumn(
        "bbox",
        expr(
            f"""
        array(
            ST_X(ST_Centroid(envelope)) - {bbox_length/2},
            ST_Y(ST_Centroid(envelope)) - {bbox_length/2},
            ST_X(ST_Centroid(envelope)) + {bbox_length/2},
            ST_Y(ST_Centroid(envelope)) + {bbox_length/2}
        )
        """
        ),
    ).drop("envelope")
    return df


def make_envelope(df: DataFrame, bbox_length: int) -> DataFrame:
    """
    Lager en minimal boks rundt polygonene.
    """
    return df.withColumn(
        "envelope",
        expr(
            f"""
                CASE 
                    WHEN ST_GeometryType(geometry) = 'ST_Point' 
                        THEN ST_Envelope(ST_Buffer(geometry, {bbox_length}))
                    ELSE ST_Envelope(geometry)
                END
                """
        ),
    )


def envelope_to_bboxes(df: DataFrame, tile_length=128) -> DataFrame:
    """
    Tar inn envelope og returnerer én bbox med gitt størrelse,
    sentrert på envelope.
    """
    df = (
        df.withColumn("xmin", expr("ST_XMin(envelope)"))
        .withColumn("ymin", expr("ST_YMin(envelope)"))
        .withColumn("xmax", expr("ST_XMax(envelope)"))
        .withColumn("ymax", expr("ST_YMax(envelope)"))
    )

    df = df.withColumn("x_center", (col("xmin") + col("xmax")) / 2.0).withColumn(
        "y_center", (col("ymin") + col("ymax")) / 2.0
    )

    df = df.withColumn("x", col("x_center") - lit(tile_length / 2.0)).withColumn(
        "y", col("y_center") - lit(tile_length / 2.0)
    )

    df = df.withColumn(
        "bbox", expr(f"array(x, y, x + {tile_length}, y + {tile_length})")
    ).withColumn("bbox_str", concat_ws(",", col("bbox")))

    drop_cols = [
        "envelope",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "x_center",
        "y_center",
        "x",
        "y",
    ]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(c)

    return df


def add_bbox_columns(df: DataFrame, bbox_length: int, max_offset: float):
    """
    Legger til mer omfattende kolonner om bbox.
    """
    adjusted_bbox_schema = StructType(
        [
            StructField("bbox", ArrayType(DoubleType())),
            StructField("bbox_str", StringType()),
        ]
    )
    adjusted_bbox_udf = udf(
        lambda envelope: random_adjusted_bbox_centered(
            envelope, bbox_length, max_offset
        ),
        adjusted_bbox_schema,
    )

    df = (
        df.withColumn("adjusted_struct", adjusted_bbox_udf(col("bbox")))
        .withColumn("Adjusted_bbox", col("adjusted_struct.bbox"))
        .withColumn("bbox_str", col("adjusted_struct.bbox_str"))
    )
    return df
