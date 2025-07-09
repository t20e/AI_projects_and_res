import torch
import torch.optim
import torch.nn as nn
from datetime import datetime


def save_checkpoint(state:dict, epochs, loss, config):
    """
    Save a model’s parameter dictionary using a deserialized state_dict.

    Parameters
    ----------
        state : pytorch state_dict
            A dictionary containing
                ex : 
                    {
                        "state_dict" : model.state_dict(),
                        "optimizer" : optimizer.state_dict()
                    }
        epochs: int
            Number of epochs that the model was trained on.
        loss: float
            Mean loss of training
    """

    # === Check if the model has already been train and get the epoch and add to it.
    prev_epoch_num = 0
    if config.CON_TRAINING:
        # Split by underscore
        parts = config.LOAD_MODEL_FILE.split('_')
        # Find 'epoch' and the next part
        for i, part in enumerate(parts):
            if part == 'epoch':
                prev_epoch_num = float(parts[i+1])
                print("TESTER:", prev_epoch_num)

    epochs += int(prev_epoch_num)

    loss = float(f"{loss:.4f}")
    img_siz = config.IMAGE_SIZE

    # === Create file name
    date_str = datetime.now().strftime("%Y-%m-%d")
    # {model_architecture}_{dataset_name}__{date}_{input_size}_epoch.pt
    file_name = f"yolo_v1_taco_D_{date_str}_EPOCH_{epochs}_LOSS_{loss}_S_{img_siz}.pt"

    # === save model
    print("\n" + "#" * 32, "\n")
    print(f"-> Saving checkpoint: ")
    torch.save(state, f"./checkpoints/{file_name}")
    print(f"Saved to {file_name}")
    print("\n" + "#" * 32, "\n")



def load_checkpoint(file_name: str, yolo: nn.Module, optimizer: torch.optim, scheduler):
    """
    Loads a pre-trained model from deserialized state_dict.

    Parameters
    ----------
        file_name: str
            The models file_name. ex: Yolo_v1_taco_448_448_2025-04-27.pt.
        yolo:
            The Yolov1 class object.
        optimizer:
            The optimizer to use.
        scheduler:
            Learning rate schedule
    Returns
    -------
        last_epoch (int): the last epoch the model was trained on.
    """
    print("\n" + "#" * 64, "\n")
    print(f"Loading Model | model_name: {file_name}")

    checkpoint_path = f"./checkpoints/{file_name}"

    checkpoint = torch.load(checkpoint_path)
    print_checkpoint(checkpoint)

    yolo.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    # Load the last epoch the model was trained on to resume from there.
    last_epoch = checkpoint["epoch"]

    print(f"\n√ Loaded model.")
    print("\n" + "#" * 64, "\n")
    return last_epoch


def print_checkpoint(checkpoint):
    """Print saved model check point data"""
    print("\n Saved Model Attributes:")
    for k, v in checkpoint.items():
        if isinstance(v, dict):
            print(f"  {k:<10} → dict with {len(v)} entries")
        elif isinstance(v, torch.Tensor):
            print(f"  {k:<10} → tensor {tuple(v.shape)}")
        else:
            print(f"  {k:<10} → {v}")