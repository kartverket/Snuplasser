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
from typing import Optional
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt



geojson_path = "snuplasser_are_FeaturesToJSO.geojson"  # bruker bekreftet at dette er riktig

# === Inputverdier: det ene bildet du lastet ned
bbox = [269024.0, 6783025.0, 269126.0, 6783102.0]  # minx, miny, maxx, maxy
image_size = [500, 500]  # bredde, høyde i piksler
bbox_str = "_".join(str(int(c)) for c in bbox)
image_path = Path("data/single_test/images") / "269024.0_6783025.0_269126.0_6783102.0.png"



# === Lag transform og rasterparametre
width, height = image_size
transform = from_bounds(*bbox, width=width, height=height)

# === Les geojson og filtrer til polygoner som overlapper bildet
gdf = gpd.read_file(geojson_path).to_crs("EPSG:25833")
tile_bounds = box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
gdf_tile = gdf[gdf.geometry.intersects(tile_bounds)]

# === Rasteriser de relevante polygonene
mask = rasterize(
    [(geom, 1) for geom in gdf_tile.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype='uint8'
)

# Lag trygt filnavn fra bbox
bbox_str = "_".join(str(int(c)) for c in bbox)  # f.eks. 298090_6773710_298190_6773810
filename = f"tile_{bbox_str}.png"

# Lag mappe + full filsti
mask_folder = Path("data/single_test/masks")
mask_folder.mkdir(parents=True, exist_ok=True)
mask_img_path = mask_folder / filename

# Lagre
Image.fromarray(mask * 255).save(str(mask_img_path))


# === Visualiser bilde og maske sammen
image = Image.open(image_path)
mask = Image.open(mask_img_path)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title("Bilde")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Maske (1 = snuplass)")
plt.tight_layout()
plt.show()