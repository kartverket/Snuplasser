import asyncio
import aiofiles
from aiohttp_retry import ExponentialRetry
import logging
from PIL import Image
from io import BytesIO
import numpy as np
from pathlib import Path
from aiohttp import ClientSession
from tqdm import tqdm

import src.config as config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RETRY_OPTIONS = ExponentialRetry(attempts=5)

def get_url(base_url: str, layers: list, bbox: list, image_size: list, format: str = "image/png", crs: str = "EPSG:25833") -> str:
    """
    Constructs a URL for fetching WMS images based on specified parameters.

    Args:
        base_url (str): The base URL for the WMS service.
        layers (list): A list of layer names to include in the map.
        bbox (list): The bounding box coordinates as [minx, miny, maxx, maxy].
        image_size (list): The image size as [width, height].
        format (str, optional): The format of the image. Defaults to "image/png".
        crs (str, optional): The Coordinate Reference System. Defaults to "EPSG:25833".

    Returns:
        str: The constructed URL for the WMS request.
    """
    layers_str = ",".join(layers)
    bbox_str = ",".join(map(str, bbox))
    width, height = image_size
    url = (f"{base_url}?VERSION=1.3.0&service=WMS&request=GetMap&Format={format}&"
           f"GetFeatureInfo=text/plain&CRS={crs}&Layers={layers_str}&BBox={bbox_str}&"
           f"width={width}&height={height}")
    return url

async def fetch_image(session, url: str, max_retries=5) -> bytes:
    """
    Asynchronously fetches image data from a URL with retry logic.
    
    Args:
        session (ClientSession): The aiohttp client session for making requests.
        url (str): URL of the image to download.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        bytes: The image data if successful, None otherwise.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                logging.error(f"Failed to fetch image: HTTP status {response.status} on attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"Exception while fetching image on attempt {attempt + 1}: {e}")
        finally:
            attempt += 1
            await asyncio.sleep(0.2)  # Backoff before retrying
    
    logging.error(f"Failed to fetch image after {max_retries} attempts.")
    return None


def process_label_image(img_bytes: bytes):
    """Process label image to check for road class presence."""
    try:
        image = Image.open(BytesIO(img_bytes)).convert("L")
        image_array = np.array(image)

        # Define road as not white (assuming roads are marked in any color but white)
        road_pixels = image_array != 255
        road_percentage = np.mean(road_pixels)

        if road_percentage >= 0.01:  # At least 5% of the image must be classified as road
            return Image.fromarray(road_pixels.astype(np.uint8) * 255), True
        else:
            return None, False
    except Exception as e:
        logging.error(f"Error processing label image: {e}")
        return None, False

async def save_image(img_bytes: bytes, path: Path):
    """Saves image bytes to the specified path."""
    async with aiofiles.open(path, 'wb') as f:
        await f.write(img_bytes)
        logging.info(f"Saved image to {path}")

async def download_and_check_label(session, url: str, path: Path, max_retries=5):
    """
    Download label image, check if it meets criteria, and save if it does.
    Retries downloading if an error occurs up to max_retries times.

    Args:
        session (ClientSession): The aiohttp client session for making requests.
        url (str): URL of the label image.
        path (Path): Path to save the label image if it meets the criteria.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        bool: True if the label was successfully downloaded and saved, False otherwise.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            label_bytes = await fetch_image(session, url)
            if label_bytes:
                label_image, has_road = process_label_image(label_bytes)
                if has_road and label_image:
                    label_image.save(path)
                    logging.info(f"Label meets criteria and is saved to {path}")
                    return True
                return False  # Image does not meet criteria, no retry needed
        except Exception as e:
            logging.error(f"Error processing label image on attempt {attempt + 1}: {e}")
        attempt += 1
        await asyncio.sleep(0.2)  # Wait a bit before retrying
        
    logging.error(f"Failed to download and check label after {max_retries} attempts.")
    return False

async def process_bbox(session, bbox, base_image_url, image_folder, preferred_image_size, delay=2.0):
    """
    Process each bbox by downloading the label and image if criteria met.
    Includes a delay before starting processing of each bbox and after each network request.

    Args:
        session: The aiohttp client session for making requests.
        bbox: The bounding box coordinates.
        base_label_url: Base URL for fetching label images.
        base_image_url: Base URL for fetching satellite images.
        label_folder: The folder path for saving label images.
        image_folder: The folder path for saving satellite images.
        preferred_image_size: The desired size of the images to download.
        delay (float, optional): The delay between requests in seconds. Default is 2.0.
    """
    image_url = get_url(base_image_url, ["ortofoto"], bbox, preferred_image_size)
    image_path = image_folder / f"image_{bbox[0]}_{bbox[1]}.png"
    
    image_bytes = await fetch_image(session, image_url)
    if image_bytes:
        await save_image(image_bytes, image_path)

async def process_bbox_sequentially(session, bbox, base_image_url, image_folder, preferred_image_size, existing_image_filenames):
    """
    Sequentially process each bbox by downloading the label and image if criteria met.
    Args:
        session (ClientSession): The aiohttp client session for making requests.
        bbox (list): The bounding box coordinates.
        base_label_url (str): Base URL for fetching label images.
        base_image_url (str): Base URL for fetching satellite images.
        label_folder (Path): The folder path for saving label images.
        image_folder (Path): The folder path for saving satellite images.
        preferred_image_size (list): The desired size of the images to download.
        delay (float): The delay between processing each bbox in seconds.
    """
    image_url = get_url(base_image_url, ["ortofoto"], bbox, preferred_image_size)

    filename = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"

    if filename in existing_image_filenames:
        logging.info(f"Skipping {filename} as it already exists")
        return

    image_path = image_folder / filename

    image_bytes = await fetch_image(session, image_url)
    if image_bytes:
        await save_image(image_bytes, image_path)


async def main():
    starting_point = config.STARTING_POINT
    ending_point = config.ENDING_POINT
    preferred_image_size = [500, 500] # Bredde, Høyde i piksler
    resolution = 0.2  # Oppløsning i meter per piksel

    bbox_size = [preferred_image_size[0] * resolution, preferred_image_size[1] * resolution]

    data_folder = Path("data").joinpath(f"{starting_point[0]}_{starting_point[1]}_{ending_point[0]}_{ending_point[1]}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}")
    image_folder = data_folder / "images"
    image_folder.mkdir(parents=True, exist_ok=True)

    base_image_url = config.BASE_IMAGE_URL

    num_images_x = int((ending_point[0] - starting_point[0]) / bbox_size[0])
    num_images_y = int((ending_point[1] - starting_point[1]) / bbox_size[1])

    bboxes = [
        [starting_point[0] + x * bbox_size[0], starting_point[1] + y * bbox_size[1],
         starting_point[0] + (x + 1) * bbox_size[0], starting_point[1] + (y + 1) * bbox_size[1]]
        for x in range(num_images_x) for y in range(num_images_y)
    ]

    # Prgress bar som viser fremdrift
    pbar = tqdm(total=len(bboxes), desc="Processing BBoxes")

    existing_image_filenames = set([x.name for x in image_folder.glob("*.png")])

    async with ClientSession() as session:
        for bbox in bboxes:
            await process_bbox_sequentially(session, bbox, base_image_url, image_folder, preferred_image_size, existing_image_filenames)
            pbar.update(1)  # Manuelt oppdatere fremdriften i progress baren

    pbar.close()  # Lukk progress baren når ferdig

if __name__ == "__main__":
    asyncio.run(main())