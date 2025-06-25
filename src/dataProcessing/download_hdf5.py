import asyncio
import aiohttp
import h5py
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from io import BytesIO
from PIL import Image
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src.config as config

GEOJSON_PATH = config.GEOJSON_PATH
BASE_IMAGE_URL = config.BASE_IMAGE_URL
BASE_DOM_URL = config.BASE_DOM_URL
IMAGE_SIZE = config.IMAGE_SIZE
HDF5_PATH = "data/snuplasser.h5"

logging.basicConfig(
    filename="failed_downloads.log",
    level=logging.WARNING,
    format="%(asctime)s - %(message)s",
)  # Logg til fil for feilmeldinger ved nedlasting av bilder


# Funksjon for å lage en bounding box rundt et polygon i GeoJSON
def make_bbox_around_polygon(geojson_path, index, buffer=10):
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:25833")
    poly = gdf.geometry.iloc[index]
    minx, miny, maxx, maxy = poly.bounds
    return [minx - buffer, miny - buffer, maxx + buffer, maxy + buffer]


# Funksjon for å lage URL for nedlasting av bilde fra Geonorge WMS
def get_url(bbox, dom=False):
    bbox_str = ",".join(map(str, bbox))
    width, height = IMAGE_SIZE
    if dom:
        return (
            f"{BASE_DOM_URL}-dom-nhm-25833?request=GetMap&Format=image/png&"
            f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=NHM_DOM_25833:skyggerelieff&"
            f"BBox={bbox_str}&width={width}&height={height}"
        )
    else:
        return (
            f"{BASE_IMAGE_URL}?VERSION=1.3.0&service=WMS&request=GetMap&Format=image/png&"
            f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=ortofoto&BBox={bbox_str}&"
            f"width={width}&height={height}"
        )


# Asynkron funksjon for å laste ned og dekode bilde eller DOM-bilde
async def download_and_decode_image(url, session, rgb=True):
    async with session.get(url) as response:
        if response.status == 200:
            img_bytes = await response.read()
            mode = "RGB" if rgb else "L"
            img = Image.open(BytesIO(img_bytes)).convert(mode)
            return np.array(img)
        else:
            msg = f"❌ Feil ved nedlasting: {url} (Statuskode: {response.status})"
            print(msg)
            logging.warning(msg)  # Skriv feilen til loggfil
            return None


# Funksjon for å generere maske fra GeoDataFrame og bounding box
def generate_mask(gdf, bbox):
    bounds = box(*bbox)
    gdf_clip = gdf[gdf.geometry.intersects(bounds)]
    if gdf_clip.empty:
        print(f"⚠️ Ingen polygon i bbox: {bbox}")
        return np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
    transform = from_bounds(*bbox, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1])
    mask = rasterize(
        [(geom, 1) for geom in gdf_clip.geometry],
        out_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0]),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask


# Hovedfunksjon for å laste ned bilder og lagre i HDF5-format
async def main():
    gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:25833")
    bbox_dict = {
        idx: make_bbox_around_polygon(GEOJSON_PATH, idx, buffer=20)
        for idx in range(len(gdf))
    }

    async with aiohttp.ClientSession() as session:
        with h5py.File(HDF5_PATH, "w") as hdf5_file:
            images_dataset = hdf5_file.create_dataset(
                "images", shape=(len(bbox_dict), *IMAGE_SIZE, 4), dtype="uint8"
            )  #  4 kanaler: RGB + DOM
            masks_dataset = hdf5_file.create_dataset(
                "masks", shape=(len(bbox_dict), *IMAGE_SIZE), dtype="uint8"
            )

            for idx, bbox in bbox_dict.items():
                print(f"🔄 Behandler bilde {idx+1}/{len(bbox_dict.items())}")
                img_url = get_url(bbox)
                dom_url = get_url(bbox, dom=True)

                img_array = await download_and_decode_image(img_url, session, rgb=True)
                dom_array = await download_and_decode_image(dom_url, session, rgb=False)
                mask_arr = generate_mask(gdf, bbox)

                if img_array is None or dom_array is None:
                    print(f"⚠️ Hopper over bilde {idx+1} med bbox {bbox} pga. feil")
                    continue

                images_dataset[idx] = np.dstack(
                    (img_array, dom_array)
                )  # Kombinerer RGB og DOM
                masks_dataset[idx] = mask_arr
                print(f"✅ Lagret data for bilde {idx+1}")


if __name__ == "__main__":
    asyncio.run(main())
