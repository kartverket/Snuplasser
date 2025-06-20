import geopandas as gpd
import fiona
from dotenv import load_dotenv
import os

# Last inn miljøvariabler fra .env-filen
load_dotenv()


def extract_turning_spaces_layers(gdb_path: str, output_path: str):
    """
    Henter ut lag med snuplasser fra en GDB-fil og lagrer dem som GeoJSON.
    Args:
        gdb_path (str): Stien til GDB-filen.
        output_path (str): Stien der GeoJSON-filen skal lagres.
    """
    layers = fiona.listlayers(gdb_path)
    target_layers = [
        layer for layer in layers if layer.lower().startswith("snuplasser_areal")
    ]

    for layer in target_layers:
        gdf = gpd.read_file(gdb_path, layer=layer, engine="fiona")
        gdf_25833 = gdf.to_crs(epsg=25833)
        gdf_25833.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    gdb_path = os.environ.get("GDB_PATH")  # Hent GDB filstien fra miljøvariabler
    output_geojson_path = "turning_spaces.geojson"  # Output GeoJSON filsti
    extract_turning_spaces_layers(gdb_path, output_geojson_path)
