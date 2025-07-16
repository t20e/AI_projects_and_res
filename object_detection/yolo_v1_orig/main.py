import torch
import torch.optim as optim


# My Modules
from configs.config_loader import load_config
from data.utils.setup_transforms import setup_transforms
from model.yolov1 import YOLOv1
from data.dataset_loader import dataset_loader
from data.voc_dataset import VOCDataset
from checkpoints.utils.checkpoint_utils import load_checkpoint
from train import train
from loss import YOLOLoss

torch.manual_seed(0)

# Get configurations
cfg = load_config("yolov1.yaml", overrides=None)
# Get transformations
transforms = setup_transforms(cfg.IMAGE_SIZE)

if __name__ == "__main__":

    # ==> Init Model.
    yolo = YOLOv1(cfg=cfg, in_channels=3).to(cfg.DEVICE)

    # ==> Init Loss
    loss_fn = YOLOLoss(cfg=cfg)

    # ==> Init Optimizer.
    optimizer = optim.Adam(yolo.parameters(), lr=cfg.LEARNING_RATE)

    # ==> Init Learning rate scheduler with a warm-up
    # TODO add scheduler and warmup
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
    if cfg.CON_TRAINING:
        #  TODO retrieve the last LEARNING RATE the pre-trained model was trained on, instead of starting over with the default config LEARNING_RATE
        #           NOTE: chatgpt says it doesnt matter as long as u load the model correclty the lr should overright default in optimizer, etc.. BUT DOUBLE CHECK
        cfg.LAST_EPOCH = load_checkpoint(
            cfg.LOAD_MODEL_FILE, yolov1=yolo, optimizer=optimizer, scheduler=scheduler
        )

    # ==> Init Dataset Loader
    loader = dataset_loader(cfg=cfg, transforms=transforms, Dataset=VOCDataset)

    # ==> Train the model
    if cfg.MODE == "train":
        train(
            cfg=cfg,
            yolo=yolo,
            loss_fn=loss_fn,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
