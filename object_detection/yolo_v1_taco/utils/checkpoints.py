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

    # Check if the model has already been train and get the epoch and add to it.
    prev_epoch_num = 0
    if config.CON_TRAINING:
        # Split by underscore
        parts = config.LOAD_MODEL_FILE.split('_')
        # Find 'epoch' and the next part
        for i, part in enumerate(parts):
            if part == 'epoch':
                prev_epoch_num = int(parts[i+1])

    epochs += prev_epoch_num
        
    # save model
    print("\n" + "#" * 32, "\n")
    print(f"-> Saving checkpoint: ")
    # TODO add loss after trainings
    date_str = datetime.now().strftime("%Y-%m-%d")
    # {model_architecture}_{dataset_name}_{input_size}_epoch_date.pt
    file_name = f"Yolo_v1_taco_448_448_epoch_{epochs}_{date_str}_loss_{loss}.pt"
    torch.save(state, f"./checkpoints/{file_name}")
    print(f"Saved to {file_name}")
    print("\n" + "#" * 32, "\n")



def load_checkpoint(file_name: str, yolo: nn.Module, optimizer: torch.optim):
    """
    Loads a model’s parameter dictionary using a deserialized state_dict

    Parameters
    ----------
        file_name: str
            The models file_name. ex: Yolo_v1_taco_448_448_2025-04-27.pt.
        yolo:
            The Yolov1 class object.
        optimizer:
            The optimizer to use.
    Returns
    -------
        Pre-trained yolo v1 model.
    """
    print("\n" + "#" * 32, "\n")

    print(f"\n-> Loading Model from checkpoint. {file_name}\n")
    checkpoint_path = f"./checkpoints/{file_name}"

    checkpoint = torch.load(checkpoint_path)
    yolo.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded model.")
    print("\n" + "#" * 32, "\n")
