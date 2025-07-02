from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def get_early_stopping(config):
    return EarlyStopping(
        monitor=config.get("monitor", "val_loss"),  # val_IoU
        mode=config.get("monitor_mode", "min"),     # "max" for IoU
        patience=config["training"].get("early_stopping_patience", 5),
        verbose=True
    )

def get_model_checkpoint(config):
    metric_name = config.get("monitor", "val_loss")  # val_IoU
    filename = f"{{epoch:02d}}-{{{metric_name}:.4f}}"
    return ModelCheckpoint(
        monitor=metric_name,  
        mode=config.get("monitor_mode", "min"),     # "max" for IoU
        save_top_k=1,
        save_weights_only=True,
        filename=filename,       
    )