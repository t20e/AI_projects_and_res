import torch


def save_checkpoint(state, filename="my_checkpoint.pt"):
    print("\n=> Saving checkpoint\n")
    torch.save(state, f"./checkpoints/{filename}")


def load_checkpoint(checkpoint, model, optimizer):
    print("\n=> Loading checkpoint\n")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])