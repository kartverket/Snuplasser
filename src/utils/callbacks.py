from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

def get_early_stopping(config):
    return EarlyStopping(
        monitor=config.get("monitor", "val_loss"),  # val_IoU
        mode=config.get("monitor_mode", "min"),     # "max" for IoU
        patience=config["training"].get("early_stopping_patience", 5),
        verbose=True
    )

def get_model_checkpoint(config):
    return ModelCheckpoint(
        monitor=config.get("monitor", "val_loss"),  # val_IoU
        mode=config.get("monitor_mode", "min"),     # "max" for IoU
        save_top_k=1,
        save_weights_only=True,
        filename="{epoch:02d}-{val_loss:.2f}"       # val_IoU:.4f
    )