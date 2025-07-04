from model.unet import UNet
from model.deeplabv3_lightning import get_deeplabv3_lightning



model_registry = {
    #"unet": UNet,
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
