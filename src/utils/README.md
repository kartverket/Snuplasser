# 🛠 `utils/` — Hjelpefunksjoner og verktøy

Denne mappen inneholder generelle verktøy som brukes på tvers av prosjektet, for eksempel logging og callbacks.

---

## Innhold

| Fil | Beskrivelse |
|-----|-------------|
| `logger.py` | Setter opp MLflowLogger for sporing av eksperimenter. |
| `callbacks.py` | Oppretter `ModelCheckpoint` og `EarlyStopping` callbacks for PyTorch Lightning. |

---

## Bruk

### Logger:

```python
from utils.logger import get_logger

logger = get_logger(model_name, config)
