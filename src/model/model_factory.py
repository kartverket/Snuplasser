from lightning.pytorch import LightningModule
from .models.deeplabv3_lightning import get_deeplabv3_lightning
from .models.unet_lightning import get_unet_lightning
from .models.deeplabv3Plus_lightning import get_deeplabv3plus_lightning_model

model_registry = {
    "unet": get_unet_lightning,
    "deeplabv3": get_deeplabv3_lightning,
    "deeplabv3plus": get_deeplabv3plus_lightning_model,
}


def get_model(
    model_name: str, config: dict, checkpoint_path: str = None
) -> LightningModule:
    """
    Returnerer modell instansiert fra config, og eventuelt lastet fra checkpoint.
    Argumenter:
        model_name (str): Navn på modellen.
        config (dict): Modellkonfigurasjon.
        checkpoint_path (str): Sti til checkpoint-fil.
    Returnerer:
        LightningModule: Modell klar til trening eller inferens.
    """
    model_name = model_name.lower()
    if model_name not in model_registry:
        raise ValueError(f"Modell '{model_name}' finnes ikke i model_registry.")

    model = model_registry[model_name](config)

    if checkpoint_path:
        model_class = model.__class__
        return model_class.load_from_checkpoint(checkpoint_path)

    return model