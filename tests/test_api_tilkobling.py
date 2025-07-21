import requests


def test_api_tilkobling():
    id = 1399118
    url = f"https://nvdbapiles.atlas.vegvesen.no/vegnett/api/v4/noder/{id}"
    headers = {
        "Accept": "application/json",
        "X-Client": "Systemet for vegobjekter",
    }
    params = {"srid": "UTM33"}

    response = requests.get(url, headers=headers, params=params)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
    assert response.status_code == 200


if __name__ == "__main__":
    test_api_tilkobling()
