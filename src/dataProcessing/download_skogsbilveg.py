import requests
import pandas as pd


def hent_skogbilveier_fra_vegnett(
    kommune_id: str, antall_per_side: int = 1000
) -> pd.DataFrame:
    url = "https://nvdbapiles-v3.atlas.vegvesen.no/vegnett/veglenkesekvenser"
    headers = {"Accept": "application/json", "X-Client": "Snuplasser"}

    params = {
        "kommune": kommune_id,
        "vegsystemreferanse": "S",
        "antall": antall_per_side,
    }

    alle_objekter = []

    while True:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        alle_objekter.extend(data.get("objekter", []))

        neste = data.get("metadata", {}).get("neste", {}).get("start")
        if not neste:
            break
        params["start"] = neste

    rows = []
    for obj in alle_objekter:
        rows.append(
            {
                "veglenkesekvensid": obj.get("veglenkesekvensid"),
                "href": obj.get("href"),
                "lengde": obj.get("lengde"),
                "kommune": kommune_id,
            }
        )

    return pd.DataFrame(rows)
