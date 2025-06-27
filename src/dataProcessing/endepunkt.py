import requests
import pandas as pd
import time


def er_ekte_endepunkt(nodeid):
    url = f"https://nvdbapiles-v3.atlas.vegvesen.no/vegnett/noder/{nodeid}"
    headers = {
        "Accept": "application/vnd.vegvesen.nvdb-v3-rev4+json",
        "X-Client": "Systemet for vegobjekter",
    }
    params = {"srid": "5973"}
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
