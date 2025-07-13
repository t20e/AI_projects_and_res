import torchvision.transforms as T
from torch.utils.data import DataLoader
from argparse import Namespace


def data_loader(config:Namespace, transforms:T, Dataset):
    """
    Loads dataset loader

    Args:
        config (argparse.Namespace): Namespace object, contains all configurations.
        transforms (torchvision.transforms): transform object to resize the bboxes and images.  Normalize image tensors.
        Dataset (Dataset): Dataset class.
    """
    dataset = Dataset(
        S=config.S,
        B=config.B,
        C=config.C,
        whichDataset=config.WHICH_DATASET,
        transforms=transforms,
    )
    if config.WHICH_DATASET == "train":
        drop_last = True
    else: # ==> When validating or testing you usually want the entire dataset.
        drop_last = False

    return DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=drop_last,
    )