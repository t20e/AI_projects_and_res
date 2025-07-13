# Loads configurations from a yaml file

from dataclasses import dataclass
import yaml


@dataclass
class YOLOConfig:

    # === Device and system
    DEVICE: int = "mps"
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    # === Hyperparameters
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-2
    BATCH_SIZE: int = 64
    WEIGHT_DECAY: int = 0

    # === Continue Training Pre-Trained Models
    CON_TRAINING: bool = False
    LOAD_MODEL_FILE: str = ""
    LAST_EPOCH: int = 0


    # === Architecture
    C: int = 20
    B: int = 2
    S: int = 7
    IOU_THRESHOLD: float = 0.6
    MIN_THRESHOLD: float = 0.5

    CELL_NODES: int = 30
    LABEL_NODES: int = 1470

    # === Dataset
    DATASET: str = ""
    IMAGE_SIZE: int = 448
    CLASS_NAMES: list[str] = None 
    NUM_IMAGES: int = 0


def load_config(file_name: str = None, overrides: dict = None) -> YOLOConfig:
    """
    Load configurations from a yaml file.

    args:
        file_name (str): The config file name. i.e. (yolov1.yaml)
        overrides (dict): Dictionary containing values to override.
    Returns:
        YOLOConfig (dataclass) object
    """
    cfg_dict = {}

    # Load from YAML
    if file_name:
        with open(f"./configs/{file_name}", "r") as f:
            cfg_dict.update(yaml.safe_load(f))

    # Apply any manual overrides (from CLI or script)
    if overrides:
        cfg_dict.update(overrides)

    return YOLOConfig(**cfg_dict)


