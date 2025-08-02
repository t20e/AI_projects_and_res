import torch
from torch.utils.data import TensorDataset, DataLoader

from configs.config_loader import YOLOConfig, load_config
from data.utils.setup_transforms import setup_transforms
from model.yolov1 import YOLOv1
from data.dataset_loader import dataset_loader
from data.voc_dataset import VOCDataset
from data.utils.setup_transforms import setup_transforms, CustomCompose


def load_few_samples_to_GPU(
    cfg: YOLOConfig,
    which_dataset: str,
    num_samples: int,
    transforms: CustomCompose,
    Dataset: VOCDataset,
    print_df: bool = True,
    batch_size: bool = 2,
):
    """
    Load a small number of samples (images, labels) directly onto the GPU once at the start of training when overfitting. This will decrease training time since dataloader wont have to call VOCDataset.__getitem__() method, which is called for every image, every epoch, every time it's needed for a batch.
    Args:
        cfg (argparse.Namespace): Namespace object, contains all configurations.
        which_dataset (str): Name of dataset to grab "VOC2012_train", "VOC2012_val" or "VOC2012_test"
        num_samples (int): The number of samples (images/labels) to load from the dataset.
        batch_size (int): The number of batches to load.
        transforms (torchvision.transforms): transform object to resize the bboxes and images.  Normalize image tensors.
        Dataset (Dataset): The Dataset class not an instance of it though.
        print_df (bool): Variable to pass to VOCDataset to print the dataframe.
    """
    # 1. Instantiate dataset
    d = Dataset(
        cfg,
        which_dataset=which_dataset,
        num_samples=num_samples,
        transforms=transforms,
        print_df=print_df,
    )

    # 2. Load all data from disk into a list of CPU tensors
    all_images = []
    all_labels = []
    for i in range(len(d)):
        # This calls __getitem__ for each sample, loading it from disk and processing it
        img, label = d[i]
        all_images.append(img)
        all_labels.append(label)

    # 3. Stack and move the entire dataset to the GPU
    all_images_gpu = torch.stack(all_images).to(cfg.DEVICE)  # .to("cpu")
    all_labels_gpu = torch.stack(all_labels).to(cfg.DEVICE)  # .to("cpu")

    # 4. Create a new TensorDataset and DataLoader using the GPU tensors
    gpu_dataset = TensorDataset(all_images_gpu, all_labels_gpu)

    train_loader = DataLoader(gpu_dataset, batch_size, shuffle=True, num_workers=0)
    return train_loader


# Test run module:
#   $          python -m utils.load_small_samples_to_GPU
def test():
    print("\n\n🚧 TESTING: load_small_batches_to_GPU module \n\n")
    cfg = load_config("config_voc_dataset.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)
    load_few_samples_to_GPU(
        cfg=cfg,
        transforms=t,
        num_samples=cfg.NUM_TRAIN_SAMPLES,
        Dataset=VOCDataset,
        which_dataset=cfg.TRAIN_DIR_NAME,
        print_df=True,
        batch_size=2,
    )


if __name__ == "__main__":
    test()
