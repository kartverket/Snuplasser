# Starting point and ending point for the map view
STARTING_POINT = [250000.0000, 6796000.0000]
ENDING_POINT = [255000.0000, 6799000.0000]

# Base URL for fetching satellite images
BASE_IMAGE_URL = "https://wms.geonorge.no/skwms1/wms.nib"
GEOJSON_PATH = "turning_spaces.geojson"

# Augmentation settings
basic_aug = {
    "flip_p": 0.5,
    "rot90_p": 0.5,
    "brightness_p": 0.3,
}

strong_aug = {
    "flip_p": 0.7,
    "rot90_p": 0.8,
    "brightness_p": 0.5,
    "shift": 0.1,
    "scale": 0.3,
    "rotate": 25,
    "ssr_p": 0.8
}

augmentation_profiles = {
    "basic": basic_aug,
    "strong": strong_aug,
}