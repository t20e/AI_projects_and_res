import torch
import torchvision.transforms as T
from argparse import Namespace
import torch.optim as optim
from yolov1 import Train
from utils.checkpoints import load_checkpoint
from yolov1 import YOLOv1


torch.manual_seed(1)

# Dataset structure for one cell -> [*classes, pc_1, x, y, w, h, pc_2, x, y, w, h]. pc_1 is probability score

# <------------- Hyperparameters/Config ------------->
config = Namespace(
    DEVICE=torch.device("mps"),  # apple silicon M series
    NUM_WORKERS=2,
    PIN_MEMORY=True,
    EPOCHS=50,
    LEARNING_RATE=2e-5, # TODO implement a scheduled learning rate
    BATCH_SIZE=64,
    WEIGHT_DECAY=0,  # TODO play with weight decay


    # load a model with weights that u have been trained to train it more
    CON_TRAINING=True,  # continue to train a model
    LOAD_MODEL_FILE="Yolo_v1_taco_448_448_epoch_50_2025-04-27_loss_275.5003.pt", #ex: "Yolo_v1_taco_448_448_epoch_50_2025-04-27.pt"

    MODE="train", # "train", "test" or "valid"
    DATASET_DIR="./data",  # root path to the dataset dir
    IMAGE_SIZE=448,


    C=18,  # how many classes in the dataset
    B=2,  # how many bounding boxes does the model predict per cell
    S=7,  # split_size, how to split the image, 7x7=49 grid cells,
    IOU_THRESHOLD=0.5,  # the iou threshold when comparing bounding boxes for NMS
    MIN_THRESHOLD=0.4,  # the minimal confidence to keep a predicted bounding box
)

# The total number of nodes that a single cell has in a label for one image, which would be the size -> [*classes, pc_1, bbox1_x_y_w_h, pc_2, bbox2_x_y_w_h]. If S=7 C=18 B=2 --> 28 nodes.
config.NUM_NODES_PER_CELL = config.C + 5 * config.B

# The total number of nodes that each label has for one image. If S=7 C=18 B=2 --> 7 * 7 * (18 + 2 * 5) = 1,372 | 7x7=49 -> 49*28 = 1,372 | the * 5 is for the second bbox in the cell -> pc_2, x, y, w, h
config.NUM_NODES_PER_IMG = config.S * config.S * (config.C + config.B * 5)


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
