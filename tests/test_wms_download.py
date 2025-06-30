import requests
import os
import getpass


SECRET_TOKEN = os.environ.get("WMS_SECRET_TOKEN")
if not SECRET_TOKEN:
    SECRET_TOKEN = getpass.getpass("Skriv inn WMS-token: ")
if not SECRET_TOKEN:
    raise ValueError("Du må oppgi en WMS-token!")

bbox = [267193.479, 6661115.098, 267418.541, 6661267.233]
bbox_str = ",".join(map(str, bbox))
url = (
    "https://wms.geonorge.no/skwms1/wms.nib?"
    f"SERVICE=WMS&VERSION=1.3.0&TICKET={SECRET_TOKEN}&REQUEST=GetMap&layers=ortofoto&STYLES=Default&"
    "CRS=EPSG:25833&"
    f"BBOX={bbox_str}&width=1024&height=1024&FORMAT=image/png"
)
r = requests.get(url)
print("Status:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
if "image" in r.headers.get("Content-Type", ""):
    with open("test_ortofoto.png", "wb") as f:
        f.write(r.content)
    print("✅ Bilde lagret som test_ortofoto.png")
else:
    print("❌ Ikke et bilde. Svar:")
    print(r.text[:500])  # Hatanın ilk 500 karakteri
