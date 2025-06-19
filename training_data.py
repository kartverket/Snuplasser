import geopandas as gpd
import os
import cv2
import numpy as np
from pathlib import Path
import aiofiles
import logging
import asyncio
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

geojson_path = "snuplasser_are_FeaturesToJSO.geojson"
gdf = gpd.read_file(geojson_path)

# Sjekk formatet
print(gdf.head())

import os
import cv2

image_folder = "data/250000.0_6796000.0_255000.0_6799000.0_0.2_500_500/images/"
image_list = os.listdir(image_folder)

# Bounding box for the entire dataset (adjust as needed)
starting_point = [269024.0, 6783025.0]
ending_point = [269126.0, 6783102.0]
# starting_point = [250700.0000, 6796000.0000]
# ending_point = [251700.0000, 6797000.0000]
preferred_image_size = [500, 500]
resolution = 0.2  # Pixels per meter

min_lon, min_lat = starting_point
max_lon, max_lat = ending_point


def latlon_to_pixel(lat, lon, image_width, image_height, bbox):
    x_pixel = int((lon - bbox[0]) / (bbox[2] - bbox[0]) * image_width)
    y_pixel = int((lat - bbox[1]) / (bbox[3] - bbox[1]) * image_height)
    y_pixel = image_height - y_pixel  # Speilvender y-koordinaten
    return x_pixel, y_pixel


async def save_image(img_bytes: bytes, path: Path):
    """Saves image bytes to the specified path."""
    async with aiofiles.open(path, "wb") as f:
        await f.write(img_bytes)
        logging.info(f"Saved image to {path}")


async def main():
    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading {image_name}")
            continue

        # Get image dimensions
        image_width, image_height = image.shape[1], image.shape[0]

        # Create a blank mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        coords = os.path.splitext(image_name)[0].split("_")

        # Extract bounding box values
        img_min_lon = float(coords[0])
        img_min_lat = float(coords[1])
        img_max_lon = float(coords[2])
        img_max_lat = float(coords[3])

        # Check for snuplasser within this image's bounds
        for _, row in gdf.iterrows():
            from shapely.geometry import box

            # Replace gpd.box with shapely's box
            image_bounds = box(img_min_lon, img_min_lat, img_max_lon, img_max_lat)

            if row.geometry.intersects(image_bounds) and not row.geometry.within(
                image_bounds
            ):

                # if row.geometry.within(image_bounds):  # Check if snuplass is inside the image

                # Convert polygon coordinates to pixel positions
                points = np.array(
                    [
                        [
                            latlon_to_pixel(
                                p[1],
                                p[0],
                                image_width,
                                image_height,
                                [img_min_lon, img_min_lat, img_max_lon, img_max_lat],
                            )
                            for p in row.geometry.exterior.coords
                        ]
                    ],
                    dtype=np.int32,
                )

                cv2.fillPoly(
                    mask, [points[0]], (255,)
                )  # Fill snuplass polygon as white

        # Save mask
        data_folder = Path("data").joinpath(
            f"{starting_point[0]}_{starting_point[1]}_{ending_point[0]}_{ending_point[1]}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}"
        )
        mask_folder = data_folder / "masks"
        mask_folder.mkdir(parents=True, exist_ok=True)

        mask_filename = image_name
        mask_path = mask_folder / mask_filename
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_path)


if __name__ == "__main__":
    asyncio.run(main())
