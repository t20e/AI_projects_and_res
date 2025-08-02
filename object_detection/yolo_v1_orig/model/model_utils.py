import torch
import torch.optim
import torch.nn as nn
from datetime import datetime
import os
from configs.config_loader import YOLOConfig

# My modules
from configs.config_loader import YOLOConfig
from model.yolov1 import YOLOv1


def save_checkpoint(state: dict, epochs, loss, cfg: YOLOConfig):
    """
    Save a model’s parameter dictionary using a deserialized state_dict so we can continue training it later.

    Args:
        state: pytorch state_dict.
        epochs (int): The number of epochs that the model was trained on.
        loss (float): The mean loss after training.
    """

    # === Check if the model has already been train, get its epoch and add to it.
    prev_epoch_num = 0
    if cfg.CON_TRAINING:
        # Split by underscore
        parts = cfg.LOAD_MODEL_FILENAME.split("_")
        # Find 'epoch' and the next part
        for i, part in enumerate(parts):
            if part == "epoch":
                prev_epoch_num = float(parts[i + 1])
                print("TESTER:", prev_epoch_num)

    epochs += int(prev_epoch_num)

    loss = float(f"{loss:.4f}")
    img_siz = cfg.IMAGE_SIZE

    # === Create file name
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{cfg.MODEL_SAVE_TO_FILENAME}_yolo_v1_dataset_{cfg.DATASET}_date_{date_str}_EPOCHS_{epochs}_LOSS_{loss}_SIZE_{img_siz}.pt"

    # === save model
    print("\n" + "#" * 32, "\n")
    print(f"-> Saving checkpoint: ")
    cwd = os.getcwd()
    path = os.path.join(cwd, "model/checkpoints")
    torch.save(state, f"{path}/{file_name}")
    print(f"Saved to {file_name}")
    print("\n" + "#" * 32, "\n")


def load_checkpoint(cfg:YOLOConfig, yolov1: YOLOv1, optimizer: torch.optim, scheduler):
    """
    Loads a models checkpoint from deserialized state_dict to continue training it.

    Args:
        cfg (YOLConfig): Project Configurations.
        yolo: Instance of the YOLOv1 class.
        optimizer: The optimizer used.
        scheduler: Learning rate scheduler.
    Returns:
        last_epoch (int): the last epoch the model was trained on.
    """
    print("\n" + "#" * 64, "\n")
    print(f"Loading model Checkpoint | model_name: {cfg.LOAD_MODEL_FILENAME}\n")

    cwd = os.getcwd()
    path = os.path.join(cwd, "model/checkpoints", cfg.LOAD_MODEL_FILENAME)
    checkpoint = torch.load(path)
    print_checkpoint(checkpoint)

    yolov1.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if cfg.USE_LR_SCHEDULER:
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Load the last epoch the model was trained on to resume from there.
    last_epoch = checkpoint["epoch"]

    print(f"\n√ Loaded model.")
    print("\n" + "#" * 64, "\n")
    return last_epoch


def print_checkpoint(checkpoint):
    """Print saved model checkpoint data"""
    print("\n Saved Model Attributes:")
    for k, v in checkpoint.items():
        if isinstance(v, dict):
            print(f"  {k:<10} → dict with {len(v)} entries")
        elif isinstance(v, torch.Tensor):
            print(f"  {k:<10} → tensor {tuple(v.shape)}")
        else:
            print(f"  {k:<10} → {v}")
