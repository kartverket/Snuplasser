import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
BRUKERID = os.getenv("GEONORGE_BRUKERID")
PASSORD = os.getenv("GEONORGE_PASSORD")
token_lifetime = 55 * 60
token_start_time = time.time()


def get_token():
    """
    Henter token fra GeoNorge og returnerer det.
    """
    url = (
        f"https://baat.geonorge.no/skbaatts/req?brukerid={BRUKERID}"
        f"&passord={PASSORD}&tjenesteid=wms.nib&retformat=s"
    )
    raw_token = requests.get(url).text.strip("`")
    return raw_token


def refresh_token_if_needed():
    """
    Henter ny token om den gamle er utløpt.
    """
    global token, token_start_time
    if time.time() - token_start_time > token_lifetime:
        print("🔄 Fornyer token...")
        token = get_token()
        token_start_time = time.time()
    return token
