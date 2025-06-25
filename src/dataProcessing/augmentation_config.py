"""
Start med en baseline (ingen augmentering) for å se hvordan modellen presterer uten noen form for dataforbedring.
Deretter kan du legge til augmenteringer gradvis for å se hvordan de påvirker ytelsen.
Prosenten av augmenteringer kan justeres basert på hvor mye variasjon du ønsker i treningsdataene.
Andelen av dataen som skal gjennomgå augmentering kan justeres med `ratio`-parameteriter i `get_train_transforms`-funksjonen.
"""

# Augmentation settings
default_aug = {
    "flip_p": 0.5,
    "rot90_p": 0.5,
    "brightness_p": 0.3,
}


augmentation_profiles = {
    "default": default_aug,
}
