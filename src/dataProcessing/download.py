import requests
import numpy as np
import geopandas as gpd
from PIL import Image
from io import BytesIO
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pathlib import Path
import asyncio
import urllib.request
import os

from src.config import IMAGE_SIZE, BASE_IMAGE_URL, BASE_DOM_URL
GEOJSON_PATH = "/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/raw_geojson/turning_spaces.geojson"
print("GEOJSON_PATH:", GEOJSON_PATH)
print("Exists?", Path(GEOJSON_PATH).exists())
SECRET_TOKEN = "" # Hemmelig
print(SECRET_TOKEN)

# Buckets (Databricks)
image_path = Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/img/")
mask_path = Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/lab/")
dom_path = Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/dom/")

image_path.mkdir(parents=True, exist_ok=True)
mask_path.mkdir(parents=True, exist_ok=True)
dom_path.mkdir(parents=True, exist_ok=True)

# === Hjelpefunksjon for URL ===
def get_url(bbox, token, dom=False):  # Token trengs bare for images
    bbox_str = ",".join(map(str, bbox))
    width, height = IMAGE_SIZE
    if dom:
        return (
            f"{BASE_DOM_URL}-dom-nhm-25833?&request=GetMap&Format=image/png&"
            f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=NHM_DOM_25833:skyggerelieff&"
            f"BBox={bbox_str}&width={width}&height={height}"
        )
    else:
        return (
            f"{BASE_IMAGE_URL}?VERSION=1.3.0&TICKET={token}&service=WMS&request=GetMap&Format=image/png&"
            f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=ortofoto&BBox={bbox_str}&"
            f"width={width}&height={height}"
        )

# === Last ned bilder ===
async def download_image(bbox, save_path, token):
    url = get_url(bbox, token, dom=False)
    try:
        def fetch_and_save():
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    data = response.read()
                    with open(save_path, "wb") as f:
                        f.write(data)
                    return True, f"✅ Lagret bilde: {save_path}"
                else:
                    return False, f"❌ Feil ved nedlasting: {response.status}"
        success, message = await asyncio.to_thread(fetch_and_save)
        print(message)
    except Exception as e:
        print(f"❌ Feil ved nedlasting: {e}")

async def download_dom_image(bbox, save_path, token):
    url = get_url(bbox, token, dom=True)
    try:
        def fetch_and_save():
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    data = response.read()
                    with open(save_path, "wb") as f:
                        f.write(data)
                    return True, f"✅ Lagret DOM-bilde: {save_path}"
                else:
                    return False, f"❌ Feil ved nedlasting av DOM: {response.status}"
        success, message = await asyncio.to_thread(fetch_and_save)
        print(message)
    except Exception as e:
        print(f"❌ Feil ved nedlasting av DOM: {e}")

# === Lag maske ===
def generate_mask(geojson_path, bbox, save_path):
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:25833")
    bounds = box(*bbox)
    gdf_clip = gdf[gdf.geometry.intersects(bounds)]
    if gdf_clip.empty:
        print(f"⚠️ Ingen polygon i bbox {bbox}")
        mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
    else:
        transform = from_bounds(*bbox, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1])
        mask = rasterize(
            [(geom, 1) for geom in gdf_clip.geometry],
            out_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0]),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
    Image.fromarray(mask * 255).save(save_path)
    print(f"✅ Lagret maske: {save_path}")

# === BBOX rundt polygon ===
def make_bbox_around_polygon(geojson_path, index, buffer=10):
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:25833")
    poly = gdf.geometry.iloc[index]
    minx, miny, maxx, maxy = poly.bounds
    return [minx - buffer, miny - buffer, maxx + buffer, maxy + buffer]

# === Main ===
async def main(token):
    if not Path(GEOJSON_PATH).exists():
        print(f"❌ GeoJSON file not found: {GEOJSON_PATH}")
        return

    gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:25833")
    bbox_dict = {
        idx: make_bbox_around_polygon(GEOJSON_PATH, idx, buffer=20)
        for idx in range(len(gdf))
    }

    for idx, bbox in bbox_dict.items():
        bbox_str = "_".join(f"{int(c)}" for c in bbox)
        img_file = image_path / f"image_{bbox_str}.png"
        mask_file = mask_path / f"mask_{bbox_str}.png"
        dom_file = dom_path / f"dom_{bbox_str}.png"

        if img_file.exists() and mask_file.exists() and dom_file.exists():
            print(f"⏭️ Hopper over {img_file.name} (allerede behandlet)")
            continue

        await download_image(bbox, img_file, SECRET_TOKEN)
        generate_mask(GEOJSON_PATH, bbox, mask_file)
        await download_dom_image(bbox, dom_file, SECRET_TOKEN)

# === Kjør ===
if __name__ == "__main__":
    if asyncio.get_event_loop().is_running():
        await main(SECRET_TOKEN)
    else:
        asyncio.run(main(SECRET_TOKEN))