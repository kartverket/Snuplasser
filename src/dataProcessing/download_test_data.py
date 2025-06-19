import asyncio
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.config as config
from src.dataProcessing.download import download_image


async def main():
    """
    Laster ned testbilder som dekker hele TEST_BBOX,
    delt opp i ruter på 100x100 meter basert på IMAGE_SIZE og RESOLUTION.
    """
    print("📸 Starter nedlasting av testbilder...")

    bbox = config.TEST_BBOX
    resolution = config.RESOLUTION  # meter per pixel (f.eks. 0.2)
    image_size = config.IMAGE_SIZE  # [width_px, height_px] (f.eks. [500, 500])

    tile_width_m = image_size[0] * resolution  # 500 * 0.2 = 100m
    tile_height_m = image_size[1] * resolution  # 100m

    xmin, ymin, xmax, ymax = bbox

    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            tile_bbox = [
                x,
                y,
                min(x + tile_width_m, xmax),
                min(y + tile_height_m, ymax),
            ]
            bbox_str = "_".join(f"{int(c)}" for c in tile_bbox)
            img_name = f"image_{bbox_str}.png"
            image_path = Path("data/images/test") / img_name
            image_path.parent.mkdir(parents=True, exist_ok=True)

            if image_path.exists():
                print(f"⏭️ Hopper over {img_name}")
            else:
                await download_image(tile_bbox, image_path)

            y += tile_height_m
        x += tile_width_m


if __name__ == "__main__":
    asyncio.run(main())
