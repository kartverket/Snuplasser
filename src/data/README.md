# 📁 `data/` – Nedlasting, behandling og forberedelse

## 📌 Formål
`data/` inneholder moduler og notebooks for hele dataflyten knyttet til trenings-, validerings- og testdata for segmenteringsmodellen.  
Dette inkluderer:

- Opprettelse og oppdatering av Delta-tabeller som beskriver datasett
- Nedlasting av ortofoto og DOM-bilder fra WMS-tjenester
- Generering av binære maskefiler fra geojson-polygoner
- Splitting av data i trenings-, validerings- og holdout-sett
- PyTorch-datasett og datamoduler for modelltrening
- Nedlasting av endepunktbilder til bruk under prediksjon/testing
- Sletting og deduplisering av lagrede bilder

---

## 📂 Mappestruktur og innhold

| 📁 Fil/mappe              | 📖 Beskrivelse |
|------------------------------|-------------|
| `dataset.py`                 | Definerer `SnuplassDataset` og funksjoner for datasplitt (inkl. støtte for tren/val/holdout). |
| `snuplass_datamodule.py`     | PyTorch Lightning `DataModule` som klargjør data for trening og validering. |
| `train/`                     | Notebooks for innhenting og klargjøring av annotert treningsdata fra GeoJSON. Oppretter Delta-tabeller (`polygons_*`, `train_silver`, `utenSnuplass_*`) med WMS-paths, ID og metadata. |
| `predict/`                   | Notebooks for innhenting og behandling av data til prediksjon. Inkluderer prosessering til `predicted_*`-tabeller for ulike størrelsesklasser. |
| `predict/predicted/`         | Notebooks for videre behandling og klassifisering av prediksjonsresultater (bronze, silver, gold). |
| `download_or_delete_data/`   | Verktøy for å laste ned eller slette datafiler fra mappene. Bruker kan velge hva som slettes eller dedupliseres. |

---

## 📥 Typisk arbeidsflyt for nedlasting og behandling

1. **Hente metadata**  
   - Bruk notebooks i `train/` eller `predict/` for å hente inn metadata fra NVDB og lagre i bronze-tabeller.

2. **Laste ned bilder**  
   - Oppdater `*_silver`-tabeller med WMS-URL-er, last ned bilder (DOM + ortofoto), og sett status til `DOWNLOADED`.

3. **Generere masker**  
   - Bruk polygondata til å lage binære maskefiler (1 = snuplass, 0 = bakgrunn).

4. **Datasplitt**  
   - Splitt datasettet i trenings-, validerings- og eventuelt holdout-sett.

5. **Trening og validering**  
   - Bruk `snuplass_datamodule.py` sammen med `dataset.py` for å mate modellen med riktig datasett.

6. **Prediksjon**  
   - Kjør notebooks i `predict/` for å laste ned og prosessere bilder til prediksjon, og lagre resultatene i `predicted_*`-tabeller.

---


