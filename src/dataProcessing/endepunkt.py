import requests
import pandas as pd
import time
import src.config as config
from pathlib import Path
import sys
import os
from src.dataProcessing.download_skogsbilveg import hent_skogsbilveier_og_noder
import getpass
import asyncio

IMAGE_DIR = Path(
    "/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/endepunkt_images"
)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DOM_DIR=Path("/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/endepunkt_dom")
DOM_DIR.mkdir(parents=True, exist_ok=True)
SECRET_TOKEN = ""

SECRET_TOKEN = os.environ.get("WMS_SECRET_TOKEN")
if not SECRET_TOKEN:
    SECRET_TOKEN = getpass.getpass("Skriv inn WMS-token: ")
if not SECRET_TOKEN:
    raise ValueError("Du må oppgi en WMS-token!")


def make_bbox_around_endepunkt(x, y, buffer_x, buffer_y):
    return [x - buffer_x, y - buffer_y, x + buffer_x, y + buffer_y]


def get_wms_url(bbox, token, dom=False):
    bbox_str = ",".join(map(str, bbox))
    width, height = config.IMAGE_SIZE

    if dom:
        BASE_DOM_URL=config.BASE_DOM_URL
        return (
            f"{BASE_DOM_URL}-dom-nhm-25833?&request=GetMap&Format=image/png&"
            f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=NHM_DOM_25833:skyggerelieff&"
            f"BBox={bbox_str}&width={width}&height={height}"
        )
    
    else:
        BASE_IMAGE_URL = config.BASE_IMAGE_URL
        return (
            f"{BASE_IMAGE_URL}?VERSION=1.3.0&TICKET={token}&service=WMS&request=GetMap&Format=image/png&"
            f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=ortofoto&BBox={bbox_str}&"
            f"width={width}&height={height}"
        )


def download_image_from_wms(wms_url, save_path):
    print(f"Henter bilde fra {wms_url}")
    response = requests.get(wms_url)
    print(f"Statuskode for bilde: {response.status_code}")
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Lagret bilde: {save_path}")
        return True
    else:
        print(f"❌ Feil ved nedlasting: {response.status_code}")
        return False


def hent_wkt_koordinater(nodeid, srid="UTM33"):
    url = f"https://nvdbapiles.atlas.vegvesen.no/vegnett/api/v4/noder/{nodeid}"
    headers = {
        "Accept": "application/json",
        "X-Client": "Systemet for vegobjekter",
    }
    params = {"srid": srid}

    print(f"Henter node {nodeid} | URL : {url} | Params:{params} ")

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Statuskode for node {nodeid}: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print(f"Data for node {nodeid}: {data}")
        porter = data.get("porter", [])
        if len(porter) == 1:
            portnummer = porter[0].get("tilkobling", {}).get("portnummer")
            er_ekte = portnummer == 1 or portnummer == 2
        else:
            er_ekte = False

        wkt = data.get("geometri", {}).get("wkt")
        if wkt and wkt.startswith("POINT Z"):
            try:
               coords = wkt.replace("POINT Z", "").replace("(", "").replace(")", "").split()
               x, y = float(coords[0]), float(coords[1])
            except Exception as ex:
                print(f" Kordinat kan ikke hentes,  {nodeid}, wkt: {wkt}, Error: {ex}")
                x, y = None, None
        else:
            print(f"WKT finnes ikke, nodeid: {nodeid}, wkt: {wkt}")
            x, y = None, None
        print(f"[{nodeid}] Tilkobling OK")
        return er_ekte, wkt, x, y
    except Exception as e:
        print(f"[{nodeid}] Feil ved henting av data: {e}")
        time.sleep(0.5)
        return False, None, None, None


def filtrer_ekte_endepunkter(df, retries=2):
    ekte_rows = []
    feilette_noder = []

    for idx, row in df.iterrows():
        nodeid = row["nodeid"]
        try:
            er_ekte, wkt, x, y = hent_wkt_koordinater(nodeid)
            
            if er_ekte and x is not None and y is not None:

                ekte_rows.append({
                    "nodeid": nodeid,
                    "wkt": wkt,
                    "x": x,
                    "y": y,
                })
            
        except Exception as e:
            print(f"[{nodeid}] Feil ved henting av data: {e}")
            feilette_noder.append(nodeid)

    # Prøv igjen på feilede noder
    for retry in range(retries):
        if not feilette_noder:
            break
        print(f"🔁 Nytt forsøk på {len(feilette_noder)} feilede noder (runde {retry+1})")

        nye_feilette = []
        for nodeid in feilette_noder:
            try:
                er_ekte, wkt, x, y = hent_wkt_koordinater(nodeid)
                if er_ekte:
                    ekte_rows.append({
                    "nodeid": nodeid,
                    "wkt": wkt,
                    "x": x,
                    "y": y,
                    })
            except Exception as e:
                nye_feilette.append(nodeid)

        feilette_noder = nye_feilette  # Siste runde feilerse, bırak

    return pd.DataFrame(ekte_rows, columns=["nodeid", "wkt", "x", "y"])


def main(token):
    df = hent_skogsbilveier_og_noder("3405")
    ekte_df = filtrer_ekte_endepunkter(df)

    image_paths = []
    dom_paths=[]

    for idx, row in ekte_df.iterrows():
        x, y = row["x"], row["y"]
        nodeid = row["nodeid"]
        print(f"[{nodeid}] Koordinater: {x}, {y}")
        if x is None or y is None:
            print(f"[{nodeid}] Koordinater mangler")
            continue
        bbox = make_bbox_around_endepunkt(x, y, buffer_x=50, buffer_y=50)


        image_url=get_wms_url(bbox,token=token, dom=False)
        image_path = IMAGE_DIR / f"endepunkt_{nodeid}.png"
        success_image=download_image_from_wms(image_url, image_path)
        image_paths.append(str(image_path) if success_image else None)

        dom_url=get_wms_url(bbox,token=token, dom=True)
        dom_path= DOM_DIR / f"endepunkt_{nodeid}.png"
        success_dom=download_image_from_wms(dom_url, dom_path)
        dom_paths.append(str(dom_path) if success_dom else None)

        

    ekte_df["image_path"] = image_paths
    ekte_df["dom_path"]=dom_paths



if __name__ == "__main__":
    try:
        if asyncio.get_event_loop().is_running():
            await main(SECRET_TOKEN)
        else:
            asyncio.run(main(SECRET_TOKEN))
    except RuntimeError:
        asyncio.run(main(SECRET_TOKEN))
    