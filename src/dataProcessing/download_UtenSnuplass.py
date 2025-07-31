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
import getpass

from src.config import IMAGE_SIZE, BASE_IMAGE_URL, BASE_DOM_URL, RESOLUTION

# === Konfigürasyon ===
GEOJSON_PATH = "/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/raw_geojson/turning_spaces.geojson"

image_path = Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/img/")
mask_path = Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/lab/")
dom_path = Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/dom/")

# Token input
SECRET_TOKEN = os.environ.get("WMS_SECRET_TOKEN")
if not SECRET_TOKEN:
    SECRET_TOKEN = getpass.getpass("Skriv inn WMS-token: ")
if not SECRET_TOKEN:
    raise ValueError("Du må oppgi en WMS-token!")

# === Fonksiyonlar ===
def tile_bbox(xmin, ymin, xmax, ymax, image_size, resolution):
    tile_w = image_size[0] * resolution
    tile_h = image_size[1] * resolution
    bboxes = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            x_end = min(x + tile_w, xmax)
            y_end = min(y + tile_h, ymax)
            bboxes.append([x, y, x_end, y_end])
            y += tile_h
        x += tile_w
    return bboxes

def get_url(bbox, token, dom=False):
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

def create_empty_mask(save_path):
    mask_array = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
    mask_img = Image.fromarray(mask_array)
    mask_img.save(save_path)

async def download_image(bbox, save_path, token, dom=False):
    url = get_url(bbox, token, dom)
    try:
        def fetch_and_save():
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    data = response.read()
                    with open(save_path, "wb") as f:
                        f.write(data)
                    return True, f"✅ Lagret bilde: {save_path.name}"
                else:
                    return False, f"❌ Feil {response.status}: {response.reason}"
        success, message = await asyncio.to_thread(fetch_and_save)
        print(message)
    except Exception as e:
        print(f"❌ Feil ved nedlasting av bilde {save_path.name}: {e}")
        return False

async def main(token):
    xmin, ymin = 169086.66 , 6560298.27
    xmax, ymax = 173520.69 , 6564854.73

    bboxes = tile_bbox(xmin, ymin, xmax, ymax, IMAGE_SIZE, RESOLUTION)

    for i, bbox in enumerate(bboxes[:10]):  # sadece ilk 10 tile
        bbox_str = "_".join(f"{int(c)}" for c in bbox)
        img_file = image_path / f"image_{bbox_str}.png"
        dom_file = dom_path / f"dom_{bbox_str}.png"
        mask_file = mask_path / f"mask_{bbox_str}.png"

        if img_file.exists() and dom_file.exists() and mask_file.exists():
            print(f"⏭️ Hopper over {img_file.name} (allerede lagret)")
            continue

        await download_image(bbox, img_file, token, dom=False)
        await download_image(bbox, dom_file, token, dom=True)
        create_empty_mask(mask_file)

if __name__ == "__main__":
    if asyncio.get_event_loop().is_running():
        await main(SECRET_TOKEN)
    else:
        asyncio.run(main(SECRET_TOKEN))
