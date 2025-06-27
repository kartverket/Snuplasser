import requests


def test_api_tilkobling():
    nodeid = 1399118
    url = f"https://nvdbapiles-v3.atlas.vegvesen.no/vegnett/noder/{nodeid}"
    headers = {
        "Accept": "application/vnd.vegvesen.nvdb-v3-rev4+json",
        "X-Client": "Systemet for vegobjekter",
    }
    params = {"srid": "5973"}

    response = requests.get(url, headers=headers, params=params)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
    assert response.status_code == 200


if __name__ == "__main__":
    test_api_tilkobling()
