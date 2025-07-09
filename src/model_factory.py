from model.deeplabv3_lightning import get_deeplabv3_lightning
from model.unet_lightning import get_unet_lightning


model_registry = {
    "unet": get_unet_lightning,
    "deeplabv3": get_deeplabv3_lightning,
}

def get_model(model_name: str, params: dict):
    """
    Henter modeller fra model-mappen.
    """
    model_name = model_name.lower()
    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not found.")
    return model_registry[model_name](params)
