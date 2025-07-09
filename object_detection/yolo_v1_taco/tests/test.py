import torch

from yolov1.dataset import Dataset
from utils.load_config import load_config
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
torch.set_printoptions(threshold=torch.inf) # shows all the values when printing tensors in jupyter notebook

config = load_config()
# <------------- Transforms ------------->
class Compose(object):
    """Apply a sequence of transforms safely on (image, bboxes)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes


class Resize(object):
    """Resize the image. No change needed for bboxes since they are normalized (0-1)."""

    def __init__(self, size):
        self.size = size  # (width, height) ex: (448,448)

    def __call__(self, img, bboxes):
        img = T.Resize(self.size)(img)
        return img, bboxes  # bboxes stay the same


class ToTensor(object):
    """Convert image to Tensor. Leave bboxes as they are."""

    def __call__(self, img, bboxes):
        img = T.ToTensor()(img)  # Automatically normalize image between 0-1
        return img, bboxes


transforms = Compose(
    # transform object to resize the bboxes and images.  Normalize image tensors.
    [
        Resize((448, 448)),  # Resize image to 448x448
        ToTensor(),  # Convert image to tensor
    ]
)

dataset = Dataset(S=config.S, B=config.B, C=config.C, mode="test", dataset_path=config.DATASET_DIR, transforms=transforms)
loader = DataLoader(
    dataset=dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=0,
    pin_memory=config.PIN_MEMORY,
    shuffle=True,
    drop_last=True
)

from yolov1.model import YOLOv1
yolo = YOLOv1(in_channels=3, S=config.S, B=config.B, C=config.C).to(config.DEVICE)

loop = tqdm(loader, leave=True)

from utils.bboxes import extract_bboxes
from yolov1.loss import YoloLoss

loss_fn = YoloLoss(config)

def forward():
    for batch_idx, (x, y) in enumerate(loop):
        # move tensors to GPU
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        # predict 
        out = yolo(x)
        # reshape output (64, 1372) -> (64, 7, 7, 28)
        out = out.view(config.BATCH_SIZE, config.S, config.S, config.NUM_NODES_PER_CELL)

        # compute Loss
        loss = loss_fn(out, y)
forward()
