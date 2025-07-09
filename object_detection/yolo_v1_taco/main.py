import torch
import torchvision.transforms as T
from argparse import Namespace
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from yolov1 import TrainPipeline
from yolov1 import YOLOv1
from yolov1.dataset import Dataset
from utils.checkpoints import load_checkpoint
from utils.load_config import load_config
from utils.data_loader import data_loader

torch.manual_seed(1)

# Dataset structure for one cell -> [*classes, pc_1, x, y, w, h, pc_2, x, y, w, h]. pc_1 & pc_2 are probability scores.

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


def main(config):
    """Root function"""

    # ==> Init model.
    yolo = YOLOv1(in_channels=3, S=config.S, B=config.B, C=config.C).to(config.DEVICE)

    # ==> Init Optimizer.
    optimizer = optim.Adam(
        yolo.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # ==> Init Learning rate scheduler with a warm-up
    warm_up = LinearLR(  # warmups help prevent exploding gradients early on.
        optimizer=optimizer, start_factor=0.1, total_iters=5
    )  # 10% of LR over first 5 epochs, then back to regular LR.

    cosine = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - 5)
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warm_up, cosine],
        milestones=[5],  # switch from warm_up to cosine after epoch 5
    )
    
    # ==> Load pre-trained model.
    if config.CON_TRAINING:
        #  TODO retrieve the last LEARNING RATE instead of starting over with the default config LEARNING_RATE
        # NOTE: chatgpt says it doesnt matter as long as u load the model correclty the lr should overright default in optimizer, etc.. BUT DOUBLE CHECK
        config.LAST_EPOCH = load_checkpoint(
            file_name=config.LOAD_MODEL_FILE,
            yolo=yolo,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    # ==> Init trainer
    loader = data_loader(config, transforms, Dataset)

    t = TrainPipeline(
        config,
        yolo,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=loader,
        mode=config.MODE,
    )
    # ==> Train model
    t.train()


if __name__ == "__main__":
    print("Running Root.")
    main(config)
