import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


# My Modules
from configs.config_loader import load_config
from data.utils.setup_transforms import setup_transforms
from model.yolov1 import YOLOv1
from data.dataset_loader import dataset_loader
from data.voc_dataset import VOCDataset
from model.model_utils import load_checkpoint
from train import train
from model.loss import YOLOLoss

torch.manual_seed(0)


# Get configurations
cfg = load_config("config_voc_dataset.yaml", overrides=None)
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
    warm_up = LinearLR(  # warmups help prevent exploding gradients early on.
        optimizer=optimizer, start_factor=0.1, total_iters=5
    )  # 10% of LR over first 5 epochs, then back to regular LR.

    cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - 5)
    if cfg.USE_SCHEDULER:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warm_up, cosine],
            milestones=[5],  # Switch from warm_up to cosine after epoch 5.
        )
    else:
        scheduler = None

    # ==> Load pre-trained model.
    if cfg.CON_TRAINING:
        cfg.LAST_EPOCH = load_checkpoint(
            cfg, yolov1=yolo, optimizer=optimizer, scheduler=scheduler
        )

    # ==> Init Dataset Loaders.
    loader = dataset_loader(
        cfg=cfg,
        which_dataset=cfg.TRAIN_DIR_NAME,
        num_samples=cfg.NUM_TRAIN_SAMPLES,
        transforms=transforms,
        Dataset=VOCDataset,
        batch_size=cfg.BATCH_SIZE,
    )

    val_loader = None
    if cfg.COMPUTE_MEAN_AVERAGE_PRECISION:
        # Load the Validation set.
        val_loader = dataset_loader(
            cfg=cfg,
            which_dataset=cfg.VALIDATION_DIR_NAME,
            num_samples=cfg.NUM_VAL_SAMPLES,
            transforms=transforms,
            Dataset=VOCDataset,
            # For validation we only need one batch.
            batch_size=cfg.VAL_BATCH_SIZE
        )

    # ==> Train the model
    if cfg.MODE == "train":
        train(
            cfg=cfg,
            yolo=yolo,
            loss_fn=loss_fn,
            loader=loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
