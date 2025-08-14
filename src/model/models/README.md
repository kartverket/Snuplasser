# 🤖 `models/` — Modellarkitekturer

Denne mappen inneholder  implementasjon av modellarkitekturer som brukes til å segmentere snuplasser i flybilder.

---

## 📂 Innhold

| 📁 Fil                      | 📖 Beskrivelse |
|------------------------------|-------------|
| `deeplabv3_lightning.py`     | Modellarkitektur for DeepLabV3 implementert med PyTorch Lightning. |
| `deeplabv3Plus_lightning.py` | Modellarkitektur for DeepLabV3+ implementert med PyTorch Lightning. |
| `unet_lightning.py`          | Modellarkitektur for UNet implementert med PyTorch Lightning. |

---

#### Alle modeller bør kunne initialiseres slik:

```python
model = model_name(model_config)
```

---