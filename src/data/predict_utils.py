import numpy as np
import pandas as pd
import geopandas as gpd
from skimage import measure
from PIL import Image
from typing import Iterator
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import wkt
from pyspark.sql.functions import *
from pyspark.sql import SparkSession


def mask_to_gdf(
    bbox: str, source_file: str, predicted_masks: str
) -> tuple[gpd.GeoDataFrame, str]:
    """
    Laster inn masken for en gitt source_file, og returnerer en GeoDataFrame med polygonene.
    """
    x_min, y_min, x_max, y_max = bbox

    mask_image = Image.open(f"{predicted_masks}/{source_file}").convert("L")
    width, height = mask_image.size
    x_res = (x_max - x_min) / width
    y_res = (y_max - y_min) / height

    mask = np.array(mask_image)
    mask_bin = (mask > 127).astype(np.uint8)

    contours = measure.find_contours(mask_bin, 0.5)
    polygons = []
    for contour in contours:
        coords = []
        for y, x in contour:
            x_coord = x_min + x * x_res
            y_coord = y_max - y * y_res
            coords.append((x_coord, y_coord))
        poly = Polygon(coords)
        if poly.is_valid:
            polygons.append(poly)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:25833")
    return gdf


def make_masks_grouped_udf(predicted_masks_path: str):
    """
    Returnerer en Pandas UDF som tar inn en DataFrame.
    """
    def masks_grouped_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in iterator:
            groups = {}
            for _, row in pdf.iterrows():
                bbox = row["bbox"]
                source_file = row["source_file"]
                photo_time = row["photo_time"]

                gdf = mask_to_gdf(bbox, source_file, predicted_masks_path)
                merged_geom = unary_union(list(gdf.geometry))
                key = (source_file, photo_time)
                groups.setdefault(key, []).append(merged_geom)

            out_rows = []
            for (source_file, photo_time), geom_list in groups.items():
                try:
                    merged_all = unary_union(geom_list)
                    if merged_all is None or merged_all.is_empty:
                        continue
                    centroid = merged_all.centroid
                    out_rows.append(
                        {
                            "source_file": source_file,
                            "photo_time": photo_time,
                            "geometry_wkt": merged_all.wkt,
                            "centroid_x": float(centroid.x),
                            "centroid_y": float(centroid.y),
                        }
                    )
                except Exception:
                    continue

            yield pd.DataFrame(out_rows)  # <-- yield, not return

    return masks_grouped_udf


def get_kommune_and_fylke(
    gold_table: DataFrame, kommune_table: str, fylke_table: str, srid=25833
):
    """
    Returnerer en Spark DataFrame med kolonner 'kommunenummer', 'kommunenavn', 'fylkesnummer', 'fylkesnavn'.
    """
    try:
        spark = SparkSession.getActiveSession()
    except:
        spark = SparkSession.builder.getOrCreate()

    predicted_gold_df = spark.read.table(gold_table).withColumn(
        "pred_geom", expr(f"ST_GeomFromWKB(geometry)")
    )
    kom_df = spark.read.table(kommune_table).withColumn(
        "kom_geom", expr(f"ST_GeomFromWKB(geometry)")
    )
    fylke_df = spark.read.table(fylke_table)

    joined_df = predicted_gold_df.join(
        kom_df, expr("ST_Intersects(kom_geom, pred_geom)")
    ).join(fylke_df, expr("fylkesnummer = substr(kommunenummer, 1, 2)"))

    return joined_df
