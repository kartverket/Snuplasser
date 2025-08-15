# 🛠 `utils/` — Hjelpefunksjoner og verktøy


## 📌 Formål
Denne mappen inneholder generelle verktøy som brukes på tvers av prosjektet, for eksempel logging og callbacks.

---

## 📂 Innhold

| 📁 Fil                | 📖 Beskrivelse |
|------------------------|----------------|
| `callbacks.py`         | Inneholder funskjoner for EarlyStopping, samt lagring av modellartifakter og prediksjoner. |
| `get_from_overview.py` | Henter data fra tabeller og splitter den opp i trening-, validering-, test-, og prediksjonsett. |
| `logger.py`            | Setter opp MLflowLogger som overvåker eksperimenter og genererer navn på kjøringer. |
| `transform.py`         | Inneholder transformasjoner for treningsdata og all annen data. |

---