import requests
import pandas as pd
import time
import src.config as config
from pathlib import Path
from tqdm.notebook import tqdm
from src.dataProcessing.download_skogsbilveg import hent_skogsbilveier_og_noder

"""
def er_ekte_endepunkt(nodeid):
    url = f"https://nvdbapiles-v3.atlas.vegvesen.no/vegnett/noder/{nodeid}"
    headers = {
        "Accept": "application/vnd.vegvesen.nvdb-v3-rev4+json",
        "X-Client": "Systemet for vegobjekter",
    }
    params = {"srid": "25833"}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    porter = data.get("porter", [])
    # time.sleep(0.2)  # For å unngå for mange forespørsler på kort tid
    return len(porter) == 1


def filtrer_ekte_endepunkter(df):
    ekte_rows = []
    for idx, row in df.iterrows():
        nodeid = row["nodeid"]
        if er_ekte_endepunkt(nodeid):
            ekte_rows.append(row)
    return pd.DataFrame(ekte_rows) 
"""


def make_bbox_around_endepunkt(x, y, buffer_x, buffer_y):
    return [x - buffer_x, y - buffer_y, x + buffer_x, y + buffer_y]


def get_wms_url(bbox):
    bbox_str = ",".join(map(str, bbox))

    width, height = config.IMAGE_SIZE
    BASE_IMAGE_URL = config.BASE_IMAGE_URL
    layer = "ortofoto"

    return (
        f"{BASE_IMAGE_URL}?"
        f"SERVICE=WMS&"
        f"VERSION=1.3.0&"
        f"REQUEST=GetMap&"
        f"layers={layer}&"
        f"STYLES=Default&"
        f"CRS=EPSG:25833&"
        f"BBOX={bbox_str}&"
        f"width=1024&"
        f"height=1024&"
        f"FORMAT=image/png"
    )


def download_image_from_wms(wms_url, save_path):
    response = requests.get(wms_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Lagret bilde: {save_path}")
        return True
    else:
        print(f"❌ Feil ved nedlasting: {response.status_code}")
        return False


def hent_wkt_koordinater(nodeid, srid="utm33"):
    url = f"https://nvdbapiles-v3.atlas.vegvesen.no/vegnett/noder/{nodeid}"
    headers = {
        "Accept": "application/vnd.vegvesen.nvdb-v3-rev4+json",
        "X-Client": "Systemet for vegobjekter",
    }
    params = {"srid": srid}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    porter = data.get("porter", [])

    if len(porter) == 1:
        portnummer = porter[0].get("tilkobling", {}).get("portnummer")
        er_ekte = portnummer == 1
    else:
        er_ekte = False

    wkt = data.get("geometri", {}).get("wkt")
    if wkt and "Z(" in wkt:

        try:
            coords = wkt.split("Z(")[1].split(")")[0].split()
            x, y = float(coords[0]), float(coords[1])
        except Exception:
            x, y = None, None
    else:
        x, y = None, None

    return er_ekte, wkt, x, y


def filtrer_ekte_endepunkter(df):
    ekte_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Henter NVDB noder"):
        nodeid = row["nodeid"]
        er_ekte, wkt, x, y = hent_wkt_koordinater(nodeid)
        if er_ekte:
            ekte_rows.append({
                "nodeid": nodeid,
                "wkt": wkt,
                "x": x,
                "y": y,
            })
        time.sleep(0.1)
    return pd.DataFrame(ekte_rows, columns=["nodeid", "wkt", "x", "y"])




def main():
    df = hent_skogsbilveier_og_noder("0301")
    ekte_df = filtrer_ekte_endepunkter(df)
    image_dir = Path("images")
    image_dir.mkdir(exist_ok=True)
    image_paths = []

    for idx, row in ekte_df.iterrows():
        x, y = row["x"], row["y"]
        nodeid = row["nodeid"]
        bbox = make_bbox_around_endepunkt(x, y, buffer_x=50, buffer_y=50)
        wms_url = get_wms_url(bbox)
        image_path = image_dir / f"endepunkt_{nodeid}.png"
        success = download_image_from_wms(wms_url, image_path)
        if success:
            image_paths.append(str(image_path))
        else:
            image_paths.append(None)

    ekte_df["image_path"] = image_paths

main()