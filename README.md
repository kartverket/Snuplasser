# 🛻 Snuplasser 🔄

## 📌 Formål
Dette prosjektet har som mål å identifisere og klassifisere snuplasser på skogsbilveier og private veier fra ortofoto.
Snuplassens diameter vil gjennom forhåndsdefinerte kategorier bestemme hvilken type kjøretøy snuplassen passer for.
Resultatet vil etter kjrøing av modellen automatisk genereres som en geopackage under */Volumes/land_topografisk-gdb_dev/external_dev/static_data/DL_SNUPLASSER/geopackages/*.

---

## 📜Innhold
1. [Modellen](#hva-koden-modeller)
2. [Kjøring](#hvordan-kjøre-modellen)
3. [Innhold](#innhold)
4. [Avhengigheter](#avhengigheter)

---

## 🧠 Hva koden modeller
Modellens oppgave er todelt:
- Modellen skal lokalisere og klassifisere snuplasser på skogsbilveier og private veier i ortofoto.
- Kategoriene for klassifisering baseres på diameter og er som følger:
    - **20-25 meter**  -> personbil
    - **25-30 meter**  -> varebil
    - **30-35 meter**  -> trailer/større transport
    - **35+ meter**    -> større militære kjøretøy

---

## 🚀 Hvordan kjøre modellen
Sett opp en treningsjobb i "Jobs & Pipelines" fra Databricks og lag to tasks med det følgende:
- Task name: Velg et passende navn for jobben den skal utføre (f. eks. train og predict)
- Type: Python script
- Source: Workspace
- Path: .../Snuplasser/src/main.py
- Compute: Velg clusteret du har tilgjengelig
- Dependent libraries: Velg ".../Snuplasser/requirements.txt"
- Parameters: 
  - For trening: ["--config",".../Snuplasser/src/train.yaml"]
  - For prediksjon: ["--config",".../Snuplasser/src/predict.yaml"]

---

## 📂 Innhold

| 📁 Fil/Mappe      | 📖 Beskrivelse |
|--------------------|----------------|
| `.env`             | Inneholder miljøvariabler, som i dette tilfellet skal være "GEONORGE_BRUKERID" og "GEONORGE_PASSORD". |
| `.gitignore`       | Innholder mapper og filer som skal ignoreres av git. |
| `requirements.txt` | Inneholder pakker med tilhørende versjoner som prosjektet er avhengig av for å kjøre. |
| `src/`             | Inneholder modellen, datanedlastning, hjelpefunksjoner, konfigurasjonsfiler, og kode for trening og prediksjon. |

---

## 🛠️ Avhengigheter
- Alt av avhengigheter finnes i prosjektets requirements.txt

---