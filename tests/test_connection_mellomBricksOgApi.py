import requests

import requests
import time

nodeids = [1465686, 1472338, 1473502] 
for nodeid in nodeids:
    url = f"https://nvdbapiles-v3.atlas.vegvesen.no/vegnett/noder/{nodeid}"
    headers = {
        "Accept": "application/vnd.vegvesen.nvdb-v3-rev4+json",
        "X-Client": "Systemet for vegobjekter"
    }
    params = {"srid": "25833"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Node {nodeid}: Status {response.status_code}")
    except Exception as e:
        print(f"Node {nodeid}: ERROR {e}")
    time.sleep(0.5)  # Burada 0.5 veya daha fazla deneyebilirsin

