# Snuplass: Modelltrening og evaluering (`src/`)

Dette er hovedmappen for kjøring av eksperimenter knyttet til deteksjon og klassifisering av snuplasser i flybilder. All kjørbar kode for modelltrening, databehandling og logging er samlet her.

---

## 🔧 Innhold

| Fil / Mappe              | Beskrivelse |
|--------------------------|-------------|
| `main.py`                | Inngangspunkt for trening og evaluering. Leser YAML-konfig og starter treningsløp. |
| `static.yaml`            | Hovedkonfigurasjonsfil for eksperimenter (modellvalg, trening, data etc.). |
| `model_factory.py`       | Returnerer riktig modell basert på navn i konfigfil. |
| `config.py`              | Håndtering av bilde størrelse, resolution. Base url's. |
| `datamodules/`           | Inneholder `SnuplassDataModule` som gir trenings- og valideringsdataloader. |
| `dataProcessing/`        | Inneholder `SnuplassDataset`, datasplitting og transformasjoner. |
| `model/`                 | Modeller (f.eks. U-Net). |
| `utils/`                 | Logging, callbacks, og annen hjelpefunksjonalitet. |

---

## 🚀 Hvordan kjøre
Sett opp en treningsjobb i "Jobs & Pipelines" fra Databricks og velg følgende under task:
- Type: Python script
- Source: Workspace
- Path: .../Snuplasser/src/main.py
- Compute: Velg den man har tilgjengelig
- Dependent libraries: Velg "requirements.txt"
- Parameters: ["--config","src/static.yaml"]







