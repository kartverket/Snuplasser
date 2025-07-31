from PIL import Image
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import geopandas as gpd
from pathlib import Path
import re
import os
from src.config import IMAGE_SIZE  
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
crs = 'EPSG:25833'
table= "`land_topografisk-gdb_dev`.`ai2025`.polygons_silver"
geojson_dir = Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/predicted_masks/")
os.makedirs(geojson_dir, exist_ok=True)



def get_bbox_from_table(row_hash: str) -> list[float]:
    df = spark.sql(f"""
        SELECT adjusted_struct.bbox AS bbox
        FROM `land_topografisk-gdb_dev`.`ai2025`.polygons_silver
        WHERE row_hash = '{row_hash}'
        LIMIT 1
    """).toPandas()

    if df.empty:
        raise ValueError(f"bbox not found for row_hash: {row_hash}")

    return df["bbox"].iloc[0]

"""
def get_latest_epoch_file(folder: Path) -> Path | None:
    epoch_files = list(folder.glob("epoch_*.png"))
    if not epoch_files:
        return None
    epoch_files_sorted = sorted(epoch_files, key=lambda f: int(re.findall(r"epoch_(\d+)", f.name)[0]))
    return epoch_files_sorted[-1] """


def mask_to_geojson(mask_path: str, output_dir: Path):
    mask_path = Path(mask_path)
    filename = mask_path.name  

    match = re.match(r"preMask_(.+)\.png", filename)
    if not match:
        raise ValueError(f"Could not extract row_hash from filename: {filename}")

    row_hash = mask_path.stem.replace("preMask_", "")
    bbox = get_bbox_from_table(row_hash)

    x_min, y_min, x_max, y_max = bbox

    mask_image = Image.open(mask_path).convert("L")
    width, height = mask_image.size  
    x_res = (x_max - x_min) / width
    y_res = (y_max - y_min) / height

    mask = np.array(mask_image)

    print(f"[DEBUG] Mask shape: {mask.shape}")
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

    output_name = mask_path.stem + ".geojson"
    output_path = output_dir / output_name

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    try:
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"GeoJSON created: {output_path} ({len(polygons)} shapes)")
    except Exception as e:
        print(f"Failed to write GeoJSON: {e}")

"""
def process_all_masks(val_predictions_path: Path):
    for folder in val_predictions_path.iterdir():
        if folder.is_dir():
            latest_mask = get_latest_epoch_file(folder)
            if latest_mask:
                try:
                    print(f"Test: {latest_mask.name}")
                    mask_to_geojson(str(latest_mask))
                    print(f"Success: {latest_mask.name}")
                except Exception as e:
                    print(f"Error: {latest_mask.name} → {e}") """
