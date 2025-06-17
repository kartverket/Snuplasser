import geopandas as gpd
from dotenv import load_dotenv
import os

load_dotenv()  # Last miljøvariabler fra .env-filen

# Les GeoDataFrame fra GDB-fil
gdf = gpd.read_file(os.environ.get("GDB_PATH"), layer="snuplasser_areal_ExportFeatures")

# Lagre GeoDataFrame som GeoJSON
gdf.to_file("turning_spaces.geojson", driver="GeoJSON")