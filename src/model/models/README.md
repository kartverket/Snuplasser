# 📦 `model/` — Modellarkitekturer

## 📌 Formål
Denne mappen inneholder  implementasjon av modellarkitekturer som brukes til å segmentere snuplasser i flybilder.

---

## 🔧 Innhold

| Fil / Mappe                    | Beskrivelse |
|--------------------------------|-------------|
| `deeplabv3_lightning.py`       | Modellarkitektur for DeepLabV3 implementert med PyTorch Lightning. |
| `deeplabv3Plus_lightning.py`   | Modellarkitektur for DeepLabV3+ implementert med PyTorch Lightning. |
| `unet_lightning.py`            | Modellarkitektur for UNet implementert med PyTorch Lightning. |

---

#### Hver modell bør kunne initialiseres slik:

```python
model = model_name(model_config)