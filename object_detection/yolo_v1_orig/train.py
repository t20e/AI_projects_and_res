import torch
import torch.optim as optim
from tqdm import tqdm

# My Modules
from configs.config_loader import YOLOConfig
from data.dataset_loader import dataset_loader
from checkpoints.utils.checkpoint_utils import save_checkpoint
from loss import YOLOLoss
from model.yolov1 import YOLOv1


def train(
    cfg: YOLOConfig,
    yolo: YOLOv1,
    loader: dataset_loader,
    loss_fn: YOLOLoss,
    optimizer: optim,
    scheduler,
):
    print("\n" + "#" * 64)
    print(f"\nTraining Model")
    print("\n" + "#" * 64)

    # If training from a checkpoint, start training from its last trained epoch to (max_epoch)
    max_epoch = cfg.LAST_EPOCH + cfg.EPOCHS
    mean_loss = 0  # Mean loss is the average loss for one epoch.

    for epoch in range(cfg.LAST_EPOCH + 1, max_epoch + 1):
        print("\n\n" + "|" + "-" * 64 + "|")
        print(f"Epoch: {epoch}/{max_epoch} | Lr = {optimizer.param_groups[0]['lr'] }")

        # TODO --> compute mean average precision look into the the todo in the res folder

        # === Helper function.
        # TODO is this mean_loss here correct? This is the last loss from the last epoch correct? It can be noted down?
        # The mean_loss from mean_loss is the average loss for the current epoch.
        mean_loss = train_one_epoch(
            cfg=cfg, loader=loader, yolo=yolo, loss_fn=loss_fn, optimizer=optimizer
        )

        # === Update Learning Rate: at the end of every epoch. Note: different learning rates need to be updated in different areas of code; example: OneCycleLR is done per-batch.
        if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
            scheduler.step

    return
    # === Save model checkpoint.
    save_checkpoint(
        state={
            "epoch": epoch,
            "model": yolo.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict(),
            "mean_loss": mean_loss,
        },
        epochs=epoch,
        loss=mean_loss,
        cfg=cfg,
    )


def train_one_epoch(cfg: YOLOConfig, loader, yolo, loss_fn, optimizer):
    """<- Helper function for each epoch. ->"""
    loop = tqdm(loader, leave=True)

    # --- Store the loss of the models bbox predictions vs true bboxes that it makes for every image.
    loss_history = []

    for batch_idx, (x, y) in enumerate(loop):
        """
        x: represents a batch of input data (images).
        y: represents the corresponding batch of ground truth labels bounding boxes for that image 'x'.
        Forward propagation:
             'x' (images) is passed through the model to get predictions. Initially, these predictions are random.
        Backpropagation:
            The model's internal parameters are adjusted based on the difference between its predictions and the true labels ('y') bounding boxes. This adjustment allows the model to progressively improve its ability to accurately predict labels.
        """
        # --- Move tensors to GPU
        x, y = x.to(cfg.DEVICE), y.to(
            cfg.DEVICE
        )  # x = (batch_size, 3, 448, 448) & y = (batch_size, 7, 7, 28)

        # === Forward-propagation | Predict
        out = yolo(x)  # (b_s, 1470)

        # --- Reshape output.
        b_s = x.size(0)  # Batch size not hardcoded for worst-case.
        out = out.view(b_s, cfg.S, cfg.S, cfg.CELL_NODES)

        # === Backward-propagation | Gradient Descent
        #   Compute loss
        loss = loss_fn(out, y)
        loss_history.append(loss.item())

        #   Clear old gradients from the previous step batch otherwise they'd accumulate.
        optimizer.zero_grad()
        #   Compute gradients of the loss w.r.t. model parameters (via backpropagation) i.e -> Gradient Descent.
        loss.backward()
        optimizer.step()  # Update the model’s parameters using the computed gradients

        #   Update the progress bar
        loop.set_postfix(loss=loss.item())

    # --- Calculate the mean loss for the epoch
    epoch_mean_loss = sum(loss_history) / len(loss_history)
    print(f"Mean loss: {epoch_mean_loss}")

    return epoch_mean_loss
