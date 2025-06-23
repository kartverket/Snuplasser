# Bruk slank Python 3.12-bilde
FROM python:3.12-slim

# Installer systemavhengigheter for geopandas, rasterio og OpenCV
RUN apt-get update && apt-get install -y \
    g++ \
    git \
    libgdal-dev \
    python3-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Sett arbeidsmappe inne i containeren
WORKDIR /snuplasser

# Kopier hele prosjektet inn i containeren
COPY . .

# Installer Python-avhengigheter
RUN pip install --no-cache-dir -r requirements.txt

# Sett PYTHONPATH slik at src/ gjenkjennes for imports
ENV PYTHONPATH=/snuplasser

# Kjør skriptet
ENTRYPOINT ["python", "-c", "import asyncio; from src.dataProcessing.download import main; asyncio.run(main('10'))"]
