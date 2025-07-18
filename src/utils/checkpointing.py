import shutil
from pathlib import Path

#Hjelpefunksjon som kopierer den beste checkpoint til en mappe
def save_best_checkpoint(model_checkpoint, model_name)-> Path:

    best_ckpt_path = Path(model_checkpoint.best_model_path)
    models_dir=Path(__file__).parent.parent/"model"/"modelsCheckpoints"
    models_dir.mkdir(exist_ok=True)

    new_name=f"{model_name}_best.ckpt"
    destination=models_dir/new_name
    shutil.copy(best_ckpt_path, destination)

    return destination