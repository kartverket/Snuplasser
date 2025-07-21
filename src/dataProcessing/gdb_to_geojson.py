import geopandas as gpd
from pathlib import Path

GDB_PATH = ""  # Denne må settes til stien til GDB-filen

GEOJSON_PATH = Path(
    "/Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/turning_spaces.geojson"
)
GEOJSON_PATH.mkdir(parents=True, exist_ok=True)


def extract_turning_spaces_layers(gdb_path: str, output_path: str):
    """
    Henter ut lag med snuplasser fra en GDB-fil og lagrer det som GeoJSON.
    Args:
        gdb_path (str): Stien til GDB-filen.
        output_path (str): Stien der GeoJSON-filen skal lagres.
    """
    gdf = gpd.read_file(gdb_path, layer="Snuplasser_areal_N50")
    gdf_25833 = gdf.to_crs(epsg=25833)
    gdf_25833.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    extract_turning_spaces_layers(GDB_PATH, GEOJSON_PATH)