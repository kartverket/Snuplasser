import geopandas as gpd
import json
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
import tempfile

import src.config as config
from src.dataProcessing.download import generate_mask
from src.dataProcessing.download import get_url
from src.dataProcessing.download import make_bbox_around_polygon
from src.dataProcessing.download import check_data_integrity


def test_get_url_generates_correct_format():
    """
    Tester at get_url genererer en gyldig WMS URL med riktig BBox og størrelse.
    """
    bbox = [100, 200, 300, 400]
    url = get_url(bbox)
    assert "BBox=100,200,300,400" in url
    assert "width=" in url and "height=" in url
    assert url.startswith("https://")


def create_dummy_geojson(path):
    data = {
        "type": "FeatureCollection",
        "name": "dummy",
        "crs": {"type": "name", "properties": {"name": "EPSG:25833"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [596000, 7030000],
                            [596100, 7030000],
                            [596100, 7030100],
                            [596000, 7030100],
                            [596000, 7030000],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def test_generate_mask_creates_valid_mask():
    """
    Tester at generate_mask lager et gyldig maske-bilde fra en GeoJSON-fil
    og en gitt BBox.
    """
    bbox = [596000, 7030000, 596100, 7030100]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        geojson_path = tmpdir / "dummy.geojson"
        save_path = tmpdir / "mask.png"

        create_dummy_geojson(geojson_path)

        generate_mask(geojson_path, bbox, save_path)

        # Sjekk at bildet ble lagret og har riktig innhold
        mask = np.array(Image.open(save_path))
        assert mask.shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0])
        assert set(np.unique(mask).tolist()).issubset({0, 255})


def test_make_bbox_around_polygon_adds_buffer():
    """
    Tester at make_bbox_around_polygon lager en BBox som er større enn polygonet
    ved å legge til en buffer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        geojson_path = tmpdir / "dummy.geojson"
        create_dummy_geojson(geojson_path)

        # Load GeoJSON into GeoDataFrame
        gdf = gpd.read_file(geojson_path)

        index = 0
        buffer = 10
        bbox = make_bbox_around_polygon(gdf, index, buffer)

        assert len(bbox) == 4
        minx, miny, maxx, maxy = bbox
        assert maxx - minx > 100  # mer enn 100 pga buffer
        assert maxy - miny > 100  # mer enn 100 pga buffer


def test_check_data_integrity_passes_with_matching_pairs():
    """
    Tester at check_data_integrity ikke kaster feil når bilder og masker matcher.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dir = Path(tmpdir) / "images"
        mask_dir = Path(tmpdir) / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        # Lag et bilde og tilhørende maske
        img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        img.save(image_dir / "image_1_2_3_4.png")
        img.save(mask_dir / "mask_1_2_3_4.png")

        check_data_integrity(image_dir, mask_dir)


def test_check_data_integrity_fails_on_mismatch():
    """
    Tester at check_data_integrity kaster AssertionError når det er en mismatch
    mellom bilder og masker.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dir = Path(tmpdir) / "images"
        mask_dir = Path(tmpdir) / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        img.save(image_dir / "image_a.png")
        img.save(mask_dir / "mask_b.png")

        with pytest.raises(AssertionError):
            check_data_integrity(image_dir, mask_dir)
