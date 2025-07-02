import requests
import os
import getpass
from pathlib import Path

# === Config ===
IMAGE_SIZE = (1024, 1024)
BASE_IMAGE_URL = "https://wms.geonorge.no/skwms1/wms.nib"
BASE_DOM_URL = "https://wms.geonorge.no/skwms1/wms.hoyde"

# === Token Handling ===
SECRET_TOKEN = os.environ.get("WMS_SECRET_TOKEN")
if not SECRET_TOKEN:
    SECRET_TOKEN = getpass.getpass("Skriv inn WMS-token: ")
if not SECRET_TOKEN:
    raise ValueError("Du må oppgi en WMS-token!")

# === get_wms_url-funksjon ===
def get_wms_url(bbox, token, dom=False):
    bbox_str = ",".join(map(str, bbox))
    width, height = IMAGE_SIZE

    if dom:
        return (
            f"{BASE_DOM_URL}-dom-nhm-25833?"
            f"request=GetMap&Format=image/png&"
            f"GetFeatureInfo=text/plain&CRS=EPSG:25833&"
            f"Layers=NHM_DOM_25833:skyggerelieff&"
            f"BBox={bbox_str}&width={width}&height={height}"
        )
    else:
        return (
            f"{BASE_IMAGE_URL}?"
            f"SERVICE=WMS&VERSION=1.3.0&TICKET={token}&REQUEST=GetMap&layers=ortofoto&"
            f"STYLES=Default&CRS=EPSG:25833&BBOX={bbox_str}&width={width}&height={height}&FORMAT=image/png"
        )

# === Koordinater ===
bbox = [267193.479, 6661115.098, 267418.541, 6661267.233]

# === Test ortofoto ===
ortofoto_url = get_wms_url(bbox, SECRET_TOKEN, dom=False)
r1 = requests.get(ortofoto_url)
print("\n[ORTOFOTO]")
print("URL:", ortofoto_url)
print("Status:", r1.status_code)
print("Content-Type:", r1.headers.get("Content-Type"))

if "image" in r1.headers.get("Content-Type", ""):
    with open("test_ortofoto.png", "wb") as f:
        f.write(r1.content)
    print("✅ Ortophoto lagret som test_ortofoto.png")
else:
    print("❌ Ikke et ortofoto-bilde. Svar:")
    print(r1.text[:500])

# === Test DOM ===
dom_url = get_wms_url(bbox, SECRET_TOKEN, dom=True)
r2 = requests.get(dom_url)
print("\n[DOM]")
print("URL:", dom_url)
print("Status:", r2.status_code)
print("Content-Type:", r2.headers.get("Content-Type"))

if "image" in r2.headers.get("Content-Type", ""):
    with open("test_dom.png", "wb") as f:
        f.write(r2.content)
    print("✅ DOM lagret som test_dom.png")
else:
    print("❌ Ikke et DOM-bilde. Svar:")
    print(r2.text[:500])
