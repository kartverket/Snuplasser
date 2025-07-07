# 📁 Nedlasting av Data

## 📌 Formål
`dataProcessing/` inneholder moduler for nedlasting, behandling og forberedelse av trenings- og testdata til bruk i segmenteringsmodellen.

Dette inkluderer:
- Nedlasting av bilder fra WMS-tjeneste
- Generering av binære maskefiler fra geojson-polygoner
- Datasplitt til trenings- og valideringssett
- Dataset- og transformklasser for PyTorch
- Nedlasting av endepunktbilder til bruk under testing
- Sletting av alle bilder i mappene med treningsbilder og DOM-bilder

---

## 📂 Innhold i `dataProcessing/`

| Fil / mappe                | Beskrivelse                                                         |
|----------------------------|---------------------------------------------------------------------|
| `augmentation_config.py´   | Definerer konfigurasjonen for augmentasjon av dataen                |
| `dataset.py`               | Definerer `SnuplassDataset` og funksjon for datasplitt              |
| `download.py`              | Laster ned bilder og oppretter maskefiler fra geojson               |
| ´download_skogsbilveg´     | Henter skogsbilveier og alle nodene på veien                        |
| ´endepunkt.py´             | Laster ned bilder av endepunkter til bruk under testing             |
| `gdb_to_geojson.py`        | Konverterer GDB til geojson (hvis du starter fra GDB)               |
| `losses.py`                | Egne tapsfunksjoner hvis aktuelt                                    |
| ´reset_images_dom´         | Sletter alle bilder i mappene med treningsbilder og DOM-bilder      |
| `transform.py`             | Inneholder transformasjoner med `albumentations`                    |
| `visualize.py`             | Brukes for å vise bildefiler og tilhørende masker                   |

---

## 📥 Fremgangsmåte for nedlasting av data

1. **Klargjør geojson**: Sørg for at du har en geojson-fil med polygoner rundt objektene (f.eks. snuplasser).

2. **Kjør `download.py`** for å:
   - Laste ned flyfoto via WMS
   - Opprette tilhørende maskefiler automatisk
   - Lagre bilder og masker i data lake (DL)
   - Det settes en buffer for å forstørre bildeutsnittet litt rundt objektet, slik at snuplassen kommer godt med

3. **Kjør endepunkt.py** for å:
   - Hente koordinater til endepunkter via WMS
   - Finn ekte endepunkter
   - Laste ned endepunktsbilder via WMS
   - Det settes en buffer for å forstørre bildeutsnittet litt rundt objektet, slik at endepunktet kommer godt med

---

## 🧪 Testing og visualisering

- Bruk `visualize.py` for å sjekke at bilder og masker stemmer overens visuelt
- Augmentering kan forhåndsvise hvordan data transformeres under trening
- Endre ´augmentation_config.py´ for å bestemme hvordan dataen skal augmenteres

---

## ✅ Neste steg

Etter at data er lastet ned og splittet:
- Velg modellen som skal brukes i ´main.py´ og sett parametere i ´static.yaml´
- Klar til å brukes med `SnuplassDataset` i `main.py`
