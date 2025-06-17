import asyncio
import aiofiles
import geopandas as gpd
from pathlib import Path
from aiohttp import ClientSession
from PIL import Image
from io import BytesIO
import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# === Konstanter ===
GEOJSON_PATH = "snuplasser_are_FeaturesToJSO.geojson"
BASE_IMAGE_URL = "https://wms.geonorge.no/skwms1/wms.nib"
IMAGE_SIZE = [500, 500]
RESOLUTION = 0.2


# === Hjelpefunksjon for WMS URL ===
def get_url(bbox):
    bbox_str = ",".join(map(str, bbox))
    width, height = IMAGE_SIZE
    return (
        f"{BASE_IMAGE_URL}?VERSION=1.3.0&service=WMS&request=GetMap&Format=image/png&"
        f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=ortofoto&BBox={bbox_str}&"
        f"width={width}&height={height}"
    )


# === Last ned bilde fra Geonorge WMS ===
async def download_image(bbox, save_path):
    url = get_url(bbox)
    async with ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                img_bytes = await response.read()
                async with aiofiles.open(save_path, "wb") as f:
                    await f.write(img_bytes)
                    print(f"✅ Lagret bilde: {save_path}")
            else:
                print(f"❌ Feil ved nedlasting: {response.status}")


# === Lag maske fra geojson og bbox ===
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
            dtype='uint8'
        )
    Image.fromarray(mask * 255).save(save_path)
    print(f"✅ Lagret maske: {save_path}")


# === Hjelpefunksjon for å lage bbox rundt polygon ===
def make_bbox_around_polygon(geojson_path, index, buffer=10):
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:25833")
    poly = gdf.geometry.iloc[index]
    minx, miny, maxx, maxy = poly.bounds
    return [minx - buffer, miny - buffer, maxx + buffer, maxy + buffer]


# === Main ===
async def main():
    # === Definer hvilke polygoner du vil bruke:
    gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:25833")
    bbox_dict = {
        idx: make_bbox_around_polygon(GEOJSON_PATH, idx, buffer=20)
        for idx in range(len(gdf))
    }

    for idx, bbox in bbox_dict.items():
        # Filnavn
        bbox_str = "_".join(f"{int(c)}" for c in bbox)
        img_name = f"image_{bbox_str}.png"
        mask_name = f"mask_{bbox_str}.png"

        # Mapper
        image_path = Path("data/images") / img_name
        mask_path = Path("data/masks") / mask_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Last ned bilde og lag maske
        await download_image(bbox, image_path)
        generate_mask(GEOJSON_PATH, bbox, mask_path)


# === Kjør
if __name__ == "__main__":
    asyncio.run(main())