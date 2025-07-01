
def get_model(model_name: str, params: dict):
    """
    Henter modeller fra model-mappen.
    """
    if model_name.lower() == "unet":
        from model.unet import UNet
        return UNet(**params)

    # elif model_name.lower() == "deeplabv3":
    #     from model.deeplabv3 import DeepLabV3
    #     return DeepLabV3(**params)
    
    

    else:
        raise Exception(f"Model {model_name} not found")