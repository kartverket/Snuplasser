VegKategori_Til_Objekttype= {
    "S": {
        "Skogbilvei": 1230  # Det 1230 er objecttype ID for Skogbilvei
              }
}

def hent_objekttype_id(vegkategori:str, navn:str)->int:
    """
    Henter objekttype ID basert på vegkategori og navn.

    Args:
        vegkategori (str): Vegkategori som skal brukes for oppslag.
        navn (str): Navn på objektet.

    Returns:
        int: Objekttype ID hvis funnet, ellers None.
    """
    kategori_data= VegKategori_Til_Objekttype.get(vegkategori.upper())
    if not kategori_data:
        raise ValueError(f"Ugyldig vegkategori: {vegkategori}")
    objekttype_id = kategori_data.get(navn)
    if objekttype_id is None:
        raise ValueError(f"Ugyldig navn for vegkategori {vegkategori}: {navn}")
    return objekttype_id
