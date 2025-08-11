#🧠SAM2

Denne mappa inneholder notebooker for sam2 med 3 og 4 kanaler og andre filer som har måttet endres for at det skulle fungere med 4 kanaler eller for å forbedre resultatene

## 📂 Innhold i `sam2/`

| Fil / mappe                | Beskrivelse                                                                                                                                                                       |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `checkpoints`              | Generer ved kjøring av en av sam2 notebookene og inneholder pretrained checkpoints fra Meta.                                                                                      |
| `configs`                  | Inneholder configs for både 3 og 4 kanaler. Hvis du ønsker å bytte checkpoint må image_encoder i treningsconfigen endres så den matcher configen tilhørende checkpointet.         |
| `sam2_logs`                | Generes under trening og et viktigste som logges her er checkpoints og litt stats fra treningen.                                                                                  |
| `augmentation.py`          | Transoformerer dataen for både 3 og 4 kanaler.                                                                                                                                    |
| `checkpoint_loader.py`     | Laster checkpoints for sam2 med 4 kanaler.                                                                                                                                        |
| `dataset_4channels.py`     | SnuplassDataset for sam2 med 4 kanaler.                                                                                                                                           |
| `dataset_3channels.py`     | SnuplassDataset for sam2 med 3 kanaler.                                                                                                                                           |
| `hiera.py`                 | Trunk for sam2 med 4 kanaler der den eneste endringen er å legge til den fjerde kanalen.                                                                                          |
| `image_predictor.py`       | Gjør prediksjoner og brukes til validering av treningen.                                                                                                                          |
| `inference.py`             | Gjør inferens for å validere treningen og plotter metrikker og artifakter til Experiments.                                                                                        |
| `loss.py`                  | Tapsfunksjonene som brukes under trening der den eneste endringen er å straffe at modellen predikerer hele bildet som en snuplass.                                                |
| `sam3_channels.ipynb`      | Notebook for å kjøre sam med 3 kanaler.                                                                                                                                           |
| `sam4_channels.ipynb`      | Notebook for å kjøre sam med 4 kanaler.                                                                                                                                           |
| `vos_dataset.py`           | Wrapper rundt SnuplassDataset-klassene for både 3 og 4 kanaler .                                                                                                                  |