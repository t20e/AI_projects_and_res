import torch

import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from yolov1 import TrainPipeline
from yolov1 import YOLOv1
from yolov1.dataset import Dataset
from utils.checkpoints import load_checkpoint
from utils.load_config import load_config
from utils.data_loader import data_loader
from utils.setup_transforms import setup_transforms

torch.manual_seed(1)

# Dataset structure for one cell -> [*classes, pc_1, x, y, w, h, pc_2, x, y, w, h]. pc_1 & pc_2 are probability scores.

# <------------- Load Hyperparameters/Config ------------->
config = load_config()



def main(config):
    """Root function"""
    transforms = setup_transforms(config.IMAGE_SIZE)

    # ==> Init model.
    yolo = YOLOv1(in_channels=3, S=config.S, B=config.B, C=config.C).to(config.DEVICE)

    # ==> Init Optimizer.
    optimizer = optim.Adam(
        yolo.parameters(), lr=config.LEARNING_RATE
    )
    # optimizer  = optim.Adam(yolo.parameters(), lr=1e-3)


    # ==> Init Learning rate scheduler with a warm-up
    # warm_up = LinearLR(  # warmups help prevent exploding gradients early on.
    #     optimizer=optimizer, start_factor=0.1, total_iters=5
    # )  # 10% of LR over first 5 epochs, then back to regular LR.

    # cosine = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - 5)
    
    scheduler = None
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warm_up, cosine],
    #     milestones=[5],  # <== switch from warm_up to cosine after epoch 5
    # )
    
    # ==> Load pre-trained model.
    if config.CON_TRAINING:
        #  TODO retrieve the last LEARNING RATE the pre-trained model was trained on, instead of starting over with the default config LEARNING_RATE
        #           NOTE: chatgpt says it doesnt matter as long as u load the model correclty the lr should overright default in optimizer, etc.. BUT DOUBLE CHECK
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
        whichDataset=config.WHICH_DATASET,
    )
    # ==> Train model
    t.train()


if __name__ == "__main__":
    print("Running Root.")
    main(config)
