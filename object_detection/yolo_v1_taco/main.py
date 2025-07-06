import torch
import torchvision.transforms as T
from argparse import Namespace
import torch.optim as optim
from yolov1 import Train
from utils.checkpoints import load_checkpoint
from yolov1 import YOLOv1
from utils.load_config import load_config

torch.manual_seed(1)

# Dataset structure for one cell -> [*classes, pc_1, x, y, w, h, pc_2, x, y, w, h]. pc_1 is probability score

# <------------- Hyperparameters/Config ------------->
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
        self.size = size  # (width, height) like (448,448)

    def __call__(self, img, bboxes):
        img = T.Resize(self.size)(img)
        return img, bboxes  # bboxes stay the same


class ToTensor(object):
    """Convert image to Tensor. Leave bboxes as they are."""

    def __call__(self, img, bboxes):
        img = T.ToTensor()(img)  # Automatically normalize image between 0-1
        return img, bboxes


transforms = Compose(
    # transform object to resize the bboxes and images.  Normalize image tensors
    [
        Resize((448, 448)),  # Resize image to 448x448
        ToTensor(),  # Convert image to tensor
    ]
)


def main(config):
    """Root function"""
    yolo = YOLOv1(in_channels=3, S=config.S, B=config.B, C=config.C).to(config.DEVICE)

    optimizer = optim.Adam(
        yolo.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    if config.CON_TRAINING:
        load_checkpoint(file_name=config.LOAD_MODEL_FILE, yolo=yolo, optimizer=optimizer)

    Train(config, yolo, optimizer=optimizer, transforms=transforms, mode=config.MODE)

if __name__ == "__main__":
    print("Running Root.")
    main(config)
