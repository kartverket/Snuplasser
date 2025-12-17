import os
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StringType,
    StructType,
    StructField,
    DateType,
)

try:
    spark = SparkSession.getActiveSession()
except:
    spark = SparkSession.builder.getOrCreate()

BASE_PATH = (
    "/Volumes/land_auto-gen-kart_dev/external_dev/static_data/DL_bildesegmentering/"
)


def generate_image_url(bbox_str, width, height):
    """
    Genererer en URL for image-bilde basert på bbox_str.
    """
    return (
        f"https://wms.geonorge.no/skwms1/wms.nib?VERSION=1.3.0"
        f"&service=WMS&request=GetMap&Format=image/png&"
        f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=ortofoto&"
        f"BBox={bbox_str}&width={width}&height={height}"
    )


def image_file_exists(id: str, image_dir: str) -> str:
    """
    Sjekker om bildet med gitt ID er lastet ned.
    """
    path = f"{BASE_PATH}{image_dir}/image_{id}.png"
    return "DOWNLOADED" if os.path.exists(path) else "PENDING"


def generate_dom_url(bbox_str, width, height):
    """
    Genererer en URL for DOM-bilde basert på bbox_str.
    """
    return (
        f"https://wms.geonorge.no/skwms1/wms.hoyde-dom-nhm-25833?request=GetMap&Format=image/png&"
        f"GetFeatureInfo=text/plain&CRS=EPSG:25833&Layers=NHM_DOM_25833:skyggerelieff&"
        f"BBOX={bbox_str}&width={width}&height={height}"
    )


def dom_file_exists(id: str, dom_dir: str) -> str:
    """
    Sjekker om DOM-bildet med gitt ID er lastet ned.
    """
    path = f"{BASE_PATH}{dom_dir}/dom_{id}.png"
    return "DOWNLOADED" if os.path.exists(path) else "PENDING"


def mask_file_exists(id: str, mask_dir: str) -> str:
    """
    Sjekker om masken med gitt ID er generert.
    """
    path = f"{BASE_PATH}{mask_dir}/mask_{id}.png"
    return "GENERATED" if os.path.exists(path) else "PENDING"


def get_fotodato(
    bbox: str, token: str, image_width: int, image_height: int, max_retries=10
):
    """
    Henter fotodato for en bbox.
    """
    url = f"https://wms.geonorge.no/skwms1/wms.nib?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&CRS=EPSG:25833&BBOX={bbox}&WIDTH={image_width}&HEIGHT={image_height}&LAYERS=ortofoto&QUERY_LAYERS=ortofoto&INFO_FORMAT=text/html&I={image_width/2}&J={image_height/2}&TICKET={token}"

    table = None
    field_value = None
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            time.sleep(1.0)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            break
        except Exception as e:
            wait = 2**i
            print(f"⚠️ Feil ved henting av fotodato ({e}), prøver igjen om {wait}s...")
            time.sleep(wait)

    if table:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2 and cells[0].text.strip() == "Fotodato":
                field_value = cells[1].text.strip()
                field_value = datetime.strptime(field_value, "%d.%m.%Y").date()
                return field_value

    return None


def add_ortofoto_date(
    df: DataFrame, token: str, id_col: str, image_width: int, image_height: int
) -> DataFrame:
    """
    Legger til fotodato kolonnen til en DataFrame.
    """
    # Henter bare relevante kolonner
    sample_rows = df.select(id_col, "bbox_str").collect()

    # Henter fotodato
    bbox_date_pairs = [
        (
            row[id_col],
            get_fotodato(
                row["bbox_str"].replace("_", ","), token, image_width, image_height
            ),
        )
        for row in sample_rows
    ]

    schema = StructType(
        [
            StructField(id_col, StringType(), True),
            StructField("photo_time", DateType(), True),
        ]
    )

    bbox_date_df = spark.createDataFrame(bbox_date_pairs, schema)
    df_with_date = df.join(bbox_date_df, on=id_col, how="left")

    return df_with_date


generate_image_url_udf = udf(generate_image_url, StringType())
image_file_exists_udf = udf(image_file_exists, StringType())
generate_dom_url_udf = udf(generate_dom_url, StringType())
dom_file_exists_udf = udf(dom_file_exists, StringType())
mask_file_exists_udf = udf(mask_file_exists, StringType())


def enrich_output(
    df: DataFrame,
    token: str,
    id_col: str,
    SUBDIR: dict,
    image_width: int,
    image_height: int,
) -> DataFrame:
    """
    Legger til kolonner med info om bilder, masker og tid.
    """
    if len(SUBDIR) == 3:
        for dt in ["image", "dom", "mask"]:
            sub = SUBDIR[dt]
            df = df.withColumn(
                f"{dt}_path",
                concat(lit(f"{BASE_PATH}/{sub}/{dt}_"), col("row_hash"), lit(".png")),
            )
        df = (
            df.withColumn(
                "dom_wms",
                generate_dom_url_udf("bbox_str", lit(image_width), lit(image_height)),
            )
            .withColumn(
                "image_wms",
                generate_image_url_udf("bbox_str", lit(image_width), lit(image_height)),
            )
            .withColumn("dom_status", dom_file_exists_udf(id_col, lit(SUBDIR["dom"])))
            .withColumn(
                "image_status", image_file_exists_udf(id_col, lit(SUBDIR["image"]))
            )
            .withColumn(
                "mask_status", mask_file_exists_udf(id_col, lit(SUBDIR["mask"]))
            )
            .withColumn("ingest_time", current_timestamp())
        )
    elif len(SUBDIR) == 2:
        if SUBDIR.get("mask") is not None:
            for dt in ["image", "mask"]:
                sub = SUBDIR[dt]
                df = df.withColumn(
                    f"{dt}_path",
                    concat(lit(f"{BASE_PATH}/{sub}/{dt}_"), col(id_col), lit(".png")),
                )
            df = (
                df.withColumn(
                    "image_wms",
                    generate_image_url_udf(
                        "bbox_str", lit(image_width), lit(image_height)
                    ),
                )
                .withColumn(
                    "image_status", image_file_exists_udf(id_col, lit(SUBDIR["image"]))
                )
                .withColumn(
                    "mask_status", mask_file_exists_udf(id_col, lit(SUBDIR["mask"]))
                )
                .withColumn("ingest_time", current_timestamp())
            )
        else:
            for dt in ["image", "dom"]:
                sub = SUBDIR[dt]
                df = df.withColumn(
                    f"{dt}_path",
                    concat(lit(f"{BASE_PATH}/{sub}/{dt}_"), col(id_col), lit(".png")),
                )
            df = (
                df.withColumn(
                    "dom_wms",
                    generate_dom_url_udf(
                        "bbox_str", lit(image_width), lit(image_height)
                    ),
                )
                .withColumn(
                    "image_wms",
                    generate_image_url_udf(
                        "bbox_str", lit(image_width), lit(image_height)
                    ),
                )
                .withColumn(
                    "dom_status", dom_file_exists_udf(id_col, lit(SUBDIR["dom"]))
                )
                .withColumn(
                    "image_status", image_file_exists_udf(id_col, lit(SUBDIR["image"]))
                )
                .withColumn("ingest_time", current_timestamp())
            )
    else:
        sub = SUBDIR["image"]
        df = df.withColumn(
            f"image_path",
            concat(lit(f"{BASE_PATH}/{sub}/iamge_"), col(id_col), lit(".png")),
        )
        df = (
            df.withColumn(
                "image_wms",
                generate_image_url_udf("bbox_str", lit(image_width), lit(image_height)),
            )
            .withColumn(
                "image_status", image_file_exists_udf(id_col, lit(SUBDIR["image"]))
            )
            .withColumn("ingest_time", current_timestamp())
        )
    df = add_ortofoto_date(df, token, id_col, image_width, image_height)
    return df
