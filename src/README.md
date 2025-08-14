# Snuplass: Modelltrening og -prediksjon (`src/`)

Dette er hovedmappen for kjøring av eksperimenter knyttet til deteksjon og klassifisering av snuplasser i flybilder. All kjørbar kode for modelltrening og -prediksjon, databehandling, modeller og logging er samlet her.

---

## 📂 Innhold

| Fil / Mappe              | Beskrivelse |
|--------------------------|-------------|
| `main.py`                | Inngangspunkt for trening og prediksjon, som leser konfigurasjonsfilen og starter trenings- eller prediksjonsløp. |
| `training.yaml`          | Hovedkonfigurasjonsfil for trening (modellvalg, treningsparametre, data etc.). |
| `predict.yaml`           | Hovedkonfigurasjonsfil for prediksjon (modellvalg, logging og data). |
| `data/`                  | Inneholder datasett, datamoduler og kode for å laste ned og slette data, samt tabeller for å trene og predikere. |
| `model/`                 | Inneholder modellene, bakgrunn for modellvalg, samt tapsfunksjonene og beregning av tapsvektene. |
| `optuna/`                | Inneholder `main_optuna` som lar deg teste modellen med flere sett av hyperparametere parallelt. |
| `utils/`                 | Inneholder transformasjonskode, callbacks, logging og koden som henter data fra tabellene. |
