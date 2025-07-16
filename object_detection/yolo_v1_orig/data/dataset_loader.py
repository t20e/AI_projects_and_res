"""
Creates a (torch.utils.data) data loader to load dataset data.
"""

from torch.utils.data import DataLoader
from data.voc_dataset import VOCDataset
from configs.config_loader import YOLOConfig, load_config
from data.utils.setup_transforms import setup_transforms, CustomCompose

def dataset_loader(cfg: YOLOConfig, transforms: CustomCompose, Dataset: VOCDataset):
    """
    Creates a dataset loader.

    Args:
        cfg (argparse.Namespace): Namespace object, contains all configurations.
        transforms (torchvision.transforms): transform object to resize the bboxes and images.  Normalize image tensors.
        Dataset (Dataset): The Dataset class not an instance of it though.
    """
    dataset = VOCDataset(cfg, transforms)

    # ==> When validating or testing you usually want the entire dataset, however when training you can drop batch_sizes that dont fit. i.e batch_size = 65, but last batch has 29 examples we drop that batch.
    if cfg.MODE == "train":
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=drop_last,
    )

# Test as module:
#  $        python -m data.dataset_loader
def test():
    print("\n\n🚧 TESTING: dataset_loader module \n\n")

    cfg = load_config("yolov1.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)

    loader = dataset_loader(cfg, t, VOCDataset)

    from tqdm import tqdm

    loop = tqdm(loader, leave=True)

    for batch_idx, (x, y) in enumerate(loop):
        print("Batch index:", batch_idx)


# The if __name__ == '__main__': guard is specifically for preventing code from being re-executed when a module is imported by another process (like the DataLoader workers).
if __name__ == '__main__':
    test()
