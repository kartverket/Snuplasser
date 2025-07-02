# 📦 `model/` — Modellarkitekturer

Denne mappen inneholder implementasjoner av modellarkitekturer som brukes til å segmentere snuplasser i flybilder.

---

## Struktur

Hver modell defineres i en egen fil, for eksempel:

- `unet.py`: U-Net-basert segmenteringsmodell
- ...

Alle modellene forventes å følge et felles grensesnitt, slik at de enkelt kan brukes via `model_factory.py`.

---

## Eksempel: Grensesnitt

Hver modell bør kunne initialiseres slik:

```python
model = UNet(**model_config)
