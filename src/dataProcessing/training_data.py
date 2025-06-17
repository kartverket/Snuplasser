import aiofiles
import asyncio
import cv2
import geopandas as gpd
import logging
import numpy as np
import os
from pathlib import Path
from PIL import Image
from shapely.geometry import box

import src.config as config

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Les geojson-fil med turning spaces
geojson_path = "turning_spaces.geojson"
gdf = gpd.read_file(geojson_path)

# Sett opp parametere for bounding box og bildeoppløsning
min_lon, min_lat = config.STARTING_POINT
max_lon, max_lat = config.ENDING_POINT
preferred_image_size = [500, 500] # Bredde, Høyde i piksler
resolution = 0.2  # Oppløsning i meter per piksel

# Sett opp mappen for lagring av bilder
image_folder = f"data/{min_lon}_{min_lat}_{max_lon}_{max_lat}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}/images/"
image_list = os.listdir(image_folder)

# Funksjon for å konvertere lat/lon til pikselkoordinater
def latlon_to_pixel(lat, lon, image_width, image_height, bbox):
    x_pixel = int((lon - bbox[0]) / (bbox[2] - bbox[0]) * image_width)
    y_pixel = int((lat - bbox[1]) / (bbox[3] - bbox[1]) * image_height)
    y_pixel = image_height - y_pixel  # Inverter y-aksen
    return x_pixel, y_pixel

# Funksjon for å lagre bildebytes til en fil
async def save_image(img_bytes: bytes, path: Path):
    async with aiofiles.open(path, 'wb') as f:
        await f.write(img_bytes)
        logging.info(f"Lagret bilde til {path}")

async def main():
    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error ved lasting av {image_name}")
            continue

        # Hent ut dimensjoner for bildet
        image_width, image_height = image.shape[1], image.shape[0]
        
        # Lag en tom maske
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Hent ut koordinater fra bilde-navnet
        coords = os.path.splitext(image_name)[0].split("_")

        # Hent ut koordinater for bildet
        img_min_lon = float(coords[0])
        img_min_lat = float(coords[1])
        img_max_lon = float(coords[2])
        img_max_lat = float(coords[3])

        # Iterer gjennom turning spaces og sjekk om de overlapper med bildet
        for _, row in gdf.iterrows():

            # Definer bildegrensene som en polygon
            image_bounds = box(img_min_lon, img_min_lat, img_max_lon, img_max_lat)

            # Sjekk om turning space polygonen overlapper med bildet
            if row.geometry.intersects(image_bounds) and not row.geometry.within(image_bounds):

                # Konverter snuplass polygonen til pikselkoordinater
                points = np.array([[
                    latlon_to_pixel(p[1], p[0], image_width, image_height, [img_min_lon, img_min_lat, img_max_lon, img_max_lat])
                    for p in row.geometry.exterior.coords
                ]], dtype=np.int32)

                cv2.fillPoly(mask, points, 255)  # Fyll polygonen i masken

        # Lagre masken som et bilde
        data_folder = Path("data").joinpath(f"{min_lon}_{min_lat}_{max_lon}_{max_lat}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}")
        mask_folder = data_folder / "masks"
        mask_folder.mkdir(parents=True, exist_ok=True)
        mask_filename = image_name
        mask_path = mask_folder / mask_filename
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_path)

if __name__ == "__main__":
    asyncio.run(main())