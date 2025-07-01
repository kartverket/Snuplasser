


model_registry = {
    "unet": UNetModel,
    #"deeplabv3": DeepLabV3Model,
}

def get_model(model_name: str, params: dict):
    """
    Henter modeller fra model-mappen.
    """
    model_name = model_name.lower()
    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not found.")
    return model_registry[model_name](**params)
