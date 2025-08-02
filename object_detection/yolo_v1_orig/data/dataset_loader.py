"""
Creates a (torch.utils.data) data loader to load dataset data.
"""

from torch.utils.data import DataLoader
from data.voc_dataset import VOCDataset
from configs.config_loader import YOLOConfig, load_config
from data.utils.setup_transforms import setup_transforms, CustomCompose


def dataset_loader(
    cfg: YOLOConfig,
    which_dataset: str,
    num_samples: int,
    transforms: CustomCompose,
    Dataset: VOCDataset,
    batch_size: int = 64,
    print_df: bool = False,
):
    """
    Creates a dataset loader.

    Args:
        cfg (argparse.Namespace): Namespace object, contains all configurations.
        which_dataset (str): Name of dataset to grab "VOC2012_train", "VOC2012_val" or "VOC2012_test"
        num_samples (int): The number of samples (images/labels) to load from the dataset.
        batch_size (int): The number of batches to load.
        transforms (torchvision.transforms): transform object to resize the bboxes and images.  Normalize image tensors.
        Dataset (Dataset): The Dataset class not an instance of it though.
        print_df (bool): Variable to pass to VOCDataset to print the dataframe.
    """
    dataset = Dataset(
        cfg=cfg,
        which_dataset=which_dataset,
        num_samples=num_samples,
        transforms=transforms,
        print_df=print_df,
    )

    if cfg.MODE == "train":
        # ==> When validating or testing you usually want the entire dataset, however when training you can drop batch_sizes that dont fit. i.e num_samples = 132, batch_size = 65 132/64 we get two batches of 64 and a batch of 4, we can drop the batch of 4.
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=drop_last,
    )


# Test as module:
#  $        python -m data.dataset_loader
def test():
    print("\n\n🚧 TESTING: dataset_loader module \n\n")

    cfg = load_config("config_voc_dataset.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)

    loader = dataset_loader(
        cfg=cfg,
        which_dataset=cfg.TRAIN_DIR_NAME,
        num_samples=cfg.NUM_TRAIN_SAMPLES,
        transforms=t,
        Dataset=VOCDataset,
        # For validation we only one batch.
        batch_size=cfg.BATCH_SIZE,
    )
    # loader = dataset_loader(
    #         cfg=cfg,
    #         which_dataset=cfg.VALIDATION_DIR_NAME,
    #         num_samples=cfg.NUM_VAL_SAMPLES,
    #         transforms=t,
    #         Dataset=VOCDataset,
    #         # For validation we only one batch.
    #         batch_size=cfg.VAL_BATCH_SIZE
    # )

    from tqdm import tqdm

    loop = tqdm(loader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        print("Batch index:", batch_idx)
        print(x.shape)


# The if __name__ == '__main__': guard is specifically for preventing code from being re-executed when a module is imported by another process (like the DataLoader workers).
if __name__ == "__main__":
    test()
