# Loads configurations from a yaml file

from dataclasses import dataclass
import yaml
import sys


@dataclass
class YOLOConfig:

    MODE: str

    # === Device and system
    DEVICE: int
    NUM_WORKERS: int
    PIN_MEMORY: bool

    # === Hyperparameters
    LEARNING_RATE: float
    EPOCHS: int = 50
    BATCH_SIZE: int = 64
    VAL_BATCH_SIZE: int = 64

    MOMENTUM: float = 0.0
    WEIGHT_DECAY: float = 0.0

    # === Continue Training Pre-Trained Models
    OVERFIT: bool = True
    SAVE_MODEL: bool = True
    CON_TRAINING: bool = False
    MODEL_SAVE_TO_FILENAME: str = ""
    LOAD_MODEL_FILENAME: str = ""
    LAST_EPOCH: int = 0

    # === Architecture
    USE_PRE_TRAIN_BACKBONE: bool = False
    C: int = 20
    B: int = 2
    S: int = 7
    NMS_IOU_THRESHOLD: float = 0.6
    NMS_MIN_THRESHOLD: float = 0.5
    mAP_IOU_THRESHOLD: float = 0.5
    USE_LR_SCHEDULER: bool = True
    CELL_NODES: int = 30
    LABEL_NODES: int = 1470
    COMPUTE_MEAN_AVERAGE_PRECISION: bool = False

    # === Dataset
    DATASET: str = ""
    IMAGE_SIZE: int = 448

    TRAIN_DIR_NAME: str = ""
    VALIDATION_DIR_NAME: str = ""
    TEST_DIR_NAME: str = ""
    ANNOTATIONS_DIR_NAME: str = ""
    IMAGES_DIR_NAME: str = ""

    CLASS_NAMES: list[str] = None
    NUM_TRAIN_SAMPLES: int = 0
    NUM_VAL_SAMPLES: int = 0


# A global flag to track if the load_config "MAJOR CONFIGURATIONS TO NOTE:" prints have already been printed.
_config_printed = False


def load_config(file_name: str = None, overrides: dict = None) -> YOLOConfig:
    """
    Load configurations from a yaml file.

    args:
        file_name (str): The config file name. i.e. (config_voc_dataset.yaml)
        overrides (dict): Dictionary containing values to override.
    Returns:
        YOLOConfig (dataclass) object
    """
    cfg_dict = {}

    # Load from YAML
    if file_name:
        with open(f"./configs/{file_name}", "r") as f:
            cfg_dict.update(yaml.safe_load(f))

    # Make sure LEARNING_RATE is a float when passing string like: '1e-6'.
    if "LEARNING_RATE" in cfg_dict:
        cfg_dict["LEARNING_RATE"] = float(cfg_dict["LEARNING_RATE"])

    # Make sure that NUM_TRAIN_SAMPLES > BATCH_SIZE
    if (
        cfg_dict["BATCH_SIZE"] > cfg_dict["NUM_TRAIN_SAMPLES"]
        and cfg_dict["NUM_TRAIN_SAMPLES"] != 0
    ):
        print("\nERROR: Config: BATCH_SIZE is greater than NUM_TRAIN_SAMPLES\n")
        sys.exit(1)

    # Do the same for NUM_VAL_SAMPLES > VAL_BATCH_SIZE
    if cfg_dict["COMPUTE_MEAN_AVERAGE_PRECISION"]:
        if (
            cfg_dict["VAL_BATCH_SIZE"] > cfg_dict["NUM_VAL_SAMPLES"]
            and cfg_dict["NUM_VAL_SAMPLES"] != 0
        ):
            print("\nERROR: Config: VAL_BATCH_SIZE is greater than NUM_VAL_SAMPLES\n")
            sys.exit(1)

    # Apply any manual overrides (from CLI or script)
    if overrides:
        cfg_dict.update(overrides)

    # If overfitting: change the configurations to perform overfitting on a couple of images.
    if cfg_dict["OVERFIT"]:
        num_samples_to_overfit = 6
        cfg_dict["NUM_WORKERS"] = 0
        # We set NUM_WORKERS=0 because the small number of samples are already on the GPU; no need for multiprocessing.
        cfg_dict["LEARNING_RATE"] = 0.0001
        cfg_dict["EPOCHS"] = 200
        cfg_dict["NUM_TRAIN_SAMPLES"] = num_samples_to_overfit
        cfg_dict["BATCH_SIZE"] = round(num_samples_to_overfit / 2)

        cfg_dict["NUM_VAL_SAMPLES"] = num_samples_to_overfit
        cfg_dict["VAL_BATCH_SIZE"] = round(num_samples_to_overfit / 2)

        cfg_dict["MODEL_SAVE_TO_FILENAME"] = (
            f"Overfit_first_{num_samples_to_overfit}_images"
        )
        cfg_dict["SAVE_MODEL"] = True
        cfg_dict["NMS_MIN_THRESHOLD"] = 0.3
        cfg_dict["NMS_IOU_THRESHOLD"] = 0.6
        cfg_dict["mAP_IOU_THRESHOLD"] = 0.0 
        cfg_dict["USE_LR_SCHEDULER"] = False # When overfitting its common better to use a set learning rate.

    # Only print if the config hasn't been printed before
    if not _config_printed:
        print("\n")
        print("#" * 64)
        print("MAJOR CONFIGURATIONS TO NOTE:")

        print("\nSystem:")
        print("\tNumber of workers:", cfg_dict["NUM_WORKERS"])
        print("\tPIN_MEMORY:", cfg_dict["PIN_MEMORY"])

        print("\nHyperparameters:")
        print("\tEPOCHS:", cfg_dict["EPOCHS"])
        print("\tLEARNING_RATE:", cfg_dict["LEARNING_RATE"])
        print("\tAmount of samples to load:")
        print(
            "\t\tNUM_TRAIN_SAMPLES:",
            cfg_dict["NUM_TRAIN_SAMPLES"],
            (
                "-> Loading all training samples."
                if cfg_dict["NUM_TRAIN_SAMPLES"] == 0
                else ""
            ),
        )
        print("\t\tBATCH_SIZE:", cfg_dict["BATCH_SIZE"])
        print("\t\tNUM_VAL_SAMPLES:", cfg_dict["NUM_VAL_SAMPLES"])
        print("\t\tVAL_BATCH_SIZE:", cfg_dict["VAL_BATCH_SIZE"])

        print("\nModel Training:")
        print("\tOVERFIT:", cfg_dict["OVERFIT"])
        print("\tCON_TRAINING:", cfg_dict["CON_TRAINING"])
        if cfg_dict["CON_TRAINING"]:
            print("\tLOAD_MODEL_FILENAME:", cfg_dict["LOAD_MODEL_FILENAME"])
        print("\tSAVE_MODEL:", cfg_dict["SAVE_MODEL"])
        print("\tMODEL_SAVE_TO_FILENAME:", cfg_dict["MODEL_SAVE_TO_FILENAME"])

        print("\nArchitecture:")
        print("\tNMS_IOU_THRESHOLD:", cfg_dict["NMS_IOU_THRESHOLD"])
        print("\tNMS_MIN_THRESHOLD:", cfg_dict["NMS_MIN_THRESHOLD"])
        print("\tmAP_IOU_THRESHOLD:", cfg_dict["mAP_IOU_THRESHOLD"])
        print("\tUSE_PRE_TRAIN_BACKBONE:", cfg_dict["USE_PRE_TRAIN_BACKBONE"])
        print("\tUSE_LR_SCHEDULER:", cfg_dict["USE_LR_SCHEDULER"])
        print(
            "\tCOMPUTE_MEAN_AVERAGE_PRECISION:",
            cfg_dict["COMPUTE_MEAN_AVERAGE_PRECISION"],
        )

        while True:  # User input
            if cfg_dict["USE_PRE_TRAIN_BACKBONE"]:
                print(
                    "\nNOTE: (USE_PRE_TRAIN_BACKBONE) is set to True, configurations have been updated for that task."
                )
            if cfg_dict["OVERFIT"]:
                print(
                    "\nNOTE: (OVERFIT) is set to true, configurations have been updated to configure it to overfit on a couple of samples!"
                )
            valid = input("\nIs the configurations correct? Type 'Y' or 'N':")
            if valid.lower() == "y":
                print("Configurations confirmed.")
                break
            elif valid.lower() == "n":
                print("Your answer was no.")
                sys.exit(1)  # Exit the script
            else:
                print("Invalid input. Please type 'Y' or 'N'.")

    return YOLOConfig(**cfg_dict)
