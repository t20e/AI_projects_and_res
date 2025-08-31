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
from utils.load_few_samples_to_memory import load_few_samples_to_memory

torch.manual_seed(0)


if __name__ == "__main__":
    # Load configurations
    cfg = load_config("config_voc_dataset.yaml", overrides=None)

    # Get transformations
    transforms = setup_transforms(cfg.IMAGE_SIZE)

    # ==> Init Model.
    if cfg.USE_PRE_TRAIN_BACKBONE:
        yolo = YOLOv1(
            cfg=cfg, in_channels=3, use_pre_trained_backbone=cfg.USE_PRE_TRAIN_BACKBONE
        ).to(cfg.DEVICE)
    else:
        yolo = YOLOv1(cfg=cfg, in_channels=3).to(cfg.DEVICE)

    # ==> Init Loss
    loss_fn = YOLOLoss(cfg=cfg)

    # ==> Init Optimizer.
    optimizer = optim.Adam(yolo.parameters(), lr=cfg.LEARNING_RATE)

    # ==> Init Learning rate scheduler with a warm-up.
    if cfg.USE_LR_SCHEDULER:
        warm_up = LinearLR(  # warmups help prevent exploding gradients early on.
            optimizer=optimizer, start_factor=0.1, total_iters=5
        )  # 10% of LR over first 5 epochs, then back to regular LR.

        cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - 5)
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
    if cfg.OVERFIT:
        # if overfit load the small number of samples onto GPU once at the start.
        print("\nImages for training")
        loader = load_few_samples_to_memory(
            cfg,
            which_dataset=cfg.OVERFIT_DIR_NAME,
            num_samples=cfg.NUM_OVERFIT_SAMPLE,
            transforms=transforms,
            Dataset=VOCDataset,
            batch_size=cfg.BATCH_SIZE,
        )
    else:
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
        if cfg.OVERFIT:
            print("Images for validation, should be the same as train when overfitting!")
            # When overfitting we want to run mAP on the same dataset as the one that we are overfitting to, so load the same train dataset for mAP.
            val_loader = load_few_samples_to_memory(
                cfg,
                which_dataset=cfg.OVERFIT_DIR_NAME,
                num_samples=cfg.NUM_OVERFIT_SAMPLE,
                transforms=transforms,
                Dataset=VOCDataset,
                batch_size=cfg.BATCH_SIZE,
            )
        else:
            # Else load the Validation.
            val_loader = dataset_loader(
                cfg=cfg,
                which_dataset=cfg.VALIDATION_DIR_NAME,
                num_samples=cfg.NUM_VAL_SAMPLES,
                transforms=transforms,
                Dataset=VOCDataset,
                # For validation we only need one batch.
                batch_size=cfg.VAL_BATCH_SIZE,
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
