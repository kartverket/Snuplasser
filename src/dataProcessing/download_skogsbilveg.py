import requests
import pandas as pd


def hent_skogsbilveier_og_noder(
    kommune_id: str, antall_per_side: int = 1000
) -> pd.DataFrame:
    url = "https://nvdbapiles-v3.atlas.vegvesen.no/vegnett/veglenkesekvenser"
    headers = {"Accept": "application/json", "X-Client": "Snuplasser"}
    params = {
        "kommune": kommune_id,
        "vegsystemreferanse": "S",
        "antall": antall_per_side,
        "inkluder": "geometri",
    }
    alle_objekter = []
    while True:
        respons = requests.get(url, headers=headers, params=params)
        respons.raise_for_status()
        data = respons.json()
        alle_objekter.extend(data.get("objekter", []))
        neste = data.get("metadata", {}).get("neste", {}).get("start")
        if not neste:
            break
        params["start"] = neste

    radliste = []
    for obj in alle_objekter:
        veglenkesekvensid = obj.get("veglenkesekvensid")
        href = obj.get("href")
        lengde = obj.get("lengde")
        porter = obj.get("porter", [])
        for p in porter:
            nodeid = p.get("tilkobling", {}).get("nodeid")
            if nodeid is not None:
                radliste.append(
                    {
                        "veglenkesekvensid": veglenkesekvensid,
                        "nodeid": nodeid,
                        "href": href,
                        "lengde": lengde,
                    }
                )

    return pd.DataFrame(radliste)
