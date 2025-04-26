import torch


def save_checkpoint(state, filename="my_checkpoint"):
    """
    Save a model’s parameter dictionary using a deserialized state_dict
    
    Parameters
    ----------
        state : pytorch state_dict
            the state_dict of a model
        filename: str
            the file name of the model.
    """
    print("\n=> Saving checkpoint\n")
    torch.save(state, f"./checkpoints/{filename}.pt")


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a model’s parameter dictionary using a deserialized state_dict
    """
    print("\n=> Loading Model from checkpoint.\n")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])