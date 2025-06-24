import asyncio
import aiofiles
import geopandas as gpd
from pathlib import Path
from aiohttp import ClientSession
from PIL import Image
import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dataProcessing.visualize import interactive_visualize
import src.config as config

"""
Modul: download.py

Denne modulen håndterer automatisert nedlasting av treningsdata til semantisk segmentering av snuplasser.

Datastrøm:
-----------
1. Inndata er en GeoJSON-fil generert fra en opprinnelig GDB (geodatabase) med polygoner som representerer snuplasser.
2. For hvert polygon beregnes en utvidet bounding box som definerer området rundt snuplassen.
3. For hver bounding box:
    - Et ortofoto lastes ned via Geonorge sin WMS-tjeneste.
    - En tilhørende binær maske genereres ved å rasterisere polygonet over bildeområdet.
4. Bildet og masken lagres som PNG-filer i henholdsvis `data/images/` og `data/masks/`.

Maskene lagres som svart-hvitt-bilder, hvor:
    - 0 (svart) representerer bakgrunn
    - 255 (hvit) representerer snuplass (dette tolkes senere som klasse 1 i modellen)

Hensikt:
--------
Modulen produserer eksakte bilde- og maskepar for hvert polygon, og gjør det mulig å bygge eksplisitt treningsdata
basert på eksisterende, manuelt annoterte snuplasser. Dette gir god kontroll over hvilke data modellen trenes på.

Bruksområde:
------------
- Forberede treningsdata til segmenteringsmodeller (f.eks. U-Net)
- Validere samsvar mellom ortofoto og eksisterende infrastruktur
- Generere dataset automatisk basert på kun geografiske polygoner
"""


# === Konstanter ===
GEOJSON_PATH = config.GEOJSON_PATH
BASE_IMAGE_URL = config.BASE_IMAGE_URL
BASE_DOM_URL = config.BASE_DOM_URL
IMAGE_SIZE = config.IMAGE_SIZE


# === Hjelpefunksjon for WMS URL ===
def get_url(bbox):
    bbox_str = ",".join(map(str, bbox))
    width, height = IMAGE_SIZE
    return (
        f"{BASE_IMAGE_URL}?VERSION=1.3.0&service=WMS&request=GetMap&Format=image/png&"
        f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=ortofoto&BBox={bbox_str}&"
        f"width={width}&height={height}"
    )


# === Hjelpefunksjon for DOM URL ===
def get_dom_url(bbox):
    bbox_str = ",".join(map(str, bbox))
    width, height = IMAGE_SIZE
    return (
        f"{BASE_DOM_URL}-dom-nhm-25833?request=GetMap&Format=image/png&"
        f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=NHM_DOM_25833:skyggerelieff&"
        f"BBox={bbox_str}&width={width}&height={height}"
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


# === Last ned DOM-bilde fra Geonorge WMS ===
async def download_dom_image(bbox, save_path):
    url = get_dom_url(bbox)
    async with ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                img_bytes = await response.read()
                async with aiofiles.open(save_path, "wb") as f:
                    await f.write(img_bytes)
                    print(f"✅ Lagret DOM-bilde: {save_path}")
            else:
                print(f"❌ Feil ved nedlasting av DOM: {response.status}")


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
            dtype="uint8",
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
        idx: make_bbox_around_polygon(
            GEOJSON_PATH, idx, buffer=20
        )  # Kan randomisere buffer for variasjon
        for idx in range(len(gdf))
    }

    for idx, bbox in bbox_dict.items():
        # Filnavn
        bbox_str = "_".join(f"{int(c)}" for c in bbox)
        img_name = f"image_{bbox_str}.png"
        mask_name = f"mask_{bbox_str}.png"
        dom_name = f"dom_{bbox_str}.png"

        # Mapper
        image_path = Path("data/images") / img_name
        mask_path = Path("data/masks") / mask_name
        dom_path = Path("data/doms") / dom_name

        # Opprett mapper hvis de ikke finnes
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        dom_path.parent.mkdir(parents=True, exist_ok=True)

        # Hopp over hvis begge finnes
        if image_path.exists() and mask_path.exists() and dom_path.exists():
            print(f"⏭️ Hopper over {img_name} (allerede lastet og masket) med DOM.")
            continue

        # Last ned bilde og DOM-bilde, og lag maske
        await download_image(bbox, image_path)
        generate_mask(GEOJSON_PATH, bbox, mask_path)
        await download_dom_image(bbox, dom_path)


def check_data_integrity(image_dir, mask_dir):
    """
    Sjekker at alle bilder har en tilhørende maske og vice versa.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    assert len(image_files) == len(
        mask_files
    ), f"Antall bilder ({len(image_files)}) og masker ({len(mask_files)}) stemmer ikke overens."

    for img, mask in zip(image_files, mask_files):
        img_id = Path(img).stem.replace("image_", "")
        mask_id = Path(mask).stem.replace("mask_", "")
        assert img_id == mask_id, f"Bildet {img} har ikke en tilhørende maske {mask}."
    print(f"✅ {len(image_files)} bilde-masker-par sjekket og verifisert.")


# === Kjør
if __name__ == "__main__":
    asyncio.run(main())
    interactive_visualize("data/images", "data/masks", "data/doms")
