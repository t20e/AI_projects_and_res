# Loads configurations from a yaml file

from dataclasses import dataclass
import yaml
import sys


@dataclass
class YOLOConfig:

    MODE: str = "train"

    # === Device and system
    DEVICE: int = "mps"
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    # === Hyperparameters
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-6
    BATCH_SIZE: int = 64
    MOMENTUM: float = 0.0
    WEIGHT_DECAY: float = 0.0

    # === Continue Training Pre-Trained Models
    CON_TRAINING: bool = False
    CUSTOM_FILE_NAME:str = ""
    LOAD_MODEL_FILE: str = ""
    LAST_EPOCH: int = 0

    # === Architecture
    C: int = 20
    B: int = 2
    S: int = 7
    IOU_THRESHOLD: float = 0.6
    MIN_THRESHOLD: float = 0.5
    USE_SCHEDULER: bool = True

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

    # Make sure LEARNING_RATE is a float when passing string: '1e-6'.
    if "LEARNING_RATE" in cfg_dict:
        cfg_dict["LEARNING_RATE"] = float(cfg_dict["LEARNING_RATE"])

    # Make sure that NUM_IMAGES > BATCH_SIZE
    if cfg_dict["BATCH_SIZE"] > cfg_dict["NUM_IMAGES"]:
        print("\nERROR: Config: BATCH_SIZE is greater than NUM_IMAGES\n")
        sys.exit(1)

    # Apply any manual overrides (from CLI or script)
    if overrides:
        cfg_dict.update(overrides)

    return YOLOConfig(**cfg_dict)
