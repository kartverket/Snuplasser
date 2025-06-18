import geopandas as gpd
import fiona
from dotenv import load_dotenv
import os

# Last inn miljøvariabler fra .env-filen
load_dotenv()

# Hent GDB filstien fra miljøvariabler
gdb_path = os.environ.get("GDB_PATH")

# Hent ut alle lag i GDB-filen
layers = fiona.listlayers(gdb_path)

# Filtrer lag for de som starter med "Snuplasser_areal"
target_layers = [layer for layer in layers if layer.lower().startswith("snuplasser_areal")]

# Iterer gjennom de filtrerte lagene og lagrer dem som GeoJSON
for layer in target_layers:
    gdf = gpd.read_file(gdb_path, layer=layer, engine="fiona")
    gdf_25833 = gdf.to_crs(epsg=25833)
    gdf_25833.to_file("turning_spaces.geojson", driver="GeoJSON")