# 📁 Nedlasting av Data

## 📌 Formål
`dataProcessing/` inneholder moduler for nedlasting, behandling og forberedelse av treningsdata til bruk i segmenteringsmodellen.

Dette inkluderer:
- Nedlasting av bilder fra WMS-tjeneste
- Generering av binære maskefiler fra geojson-polygoner
- Datasplitt til trenings- og valideringssett
- Dataset- og transformklasser for PyTorch

---

## 📂 Innhold i `dataProcessing/`

| Fil / mappe                | Beskrivelse                                                         |
|----------------------------|----------------------------------------------------------------------|
| `download.py`              | Laster ned bilder og oppretter maskefiler fra geojson               |
| `dataset.py`               | Definerer `SnuplassDataset` og funksjon for datasplitt              |
| `transform.py`             | Inneholder transformasjoner med `albumentations`                    |
| `visualize.py`             | Brukes for å vise bildefiler og tilhørende masker                   |
| `losses.py`                | Egne loss-funksjoner hvis aktuelt                                   |
| `gdb_to_geojson.py`        | Konverterer GDB til geojson (hvis du starter fra GDB)               |
| `download_test_data.py`    | Laster ned testbilder som dekker hele `TEST_BBOX`                   |

---

## 📥 Fremgangsmåte for nedlasting av data

1. **Klargjør geojson**: Sørg for at du har en geojson-fil med polygoner rundt objektene (f.eks. snuplasser).

2. **Kjør `download.py`** for å:
   - Laste ned flyfoto via WMS
   - Opprette tilhørende maskefiler automatisk
   - Lagre bilder i `data/images/` og masker i `data/masks/`

3. **Buffer rundt polygon**: Det settes en buffer for å forstørre bildeutsnittet litt rundt objektet, slik at snuplassen kommer godt med.

4. **Kjør datasplitt**: 
   ```bash
   python src/dataProcessing/dataset.py
   ```
   Dette lager `train.txt` og `val.txt` i `data/splits/`, basert på valgt `split_ratio` og `seed`.

---

## 🧪 Testing og visualisering

- Bruk `visualize.py` for å sjekke at bilder og masker stemmer overens visuelt
- Augmentering kan forhåndsvise hvordan data transformeres under trening

---

## ✅ Neste steg

Etter at data er lastet ned og splittet:
- Klar til å brukes med `SnuplassDataset` i `train.py`
- Se `split_meta.json` for info om hvordan datasplitten ble laget
