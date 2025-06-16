import geopandas as gpd
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Print value to check if it's loaded correctly
print(os.getenv("GDB_PATH"))  

# Load the GDB file
gdf = gpd.read_file(os.environ.get("GDB_PATH"), layer="snuplasser_areal_ExportFeatures")

# Save as GeoJSON
gdf.to_file("turning_spaces.geojson", driver="GeoJSON")