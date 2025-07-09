import torch
from tqdm import tqdm
from argparse import Namespace
import torch.nn as nn
import torch.optim

from .loss import YoloLoss
from utils.checkpoints import save_checkpoint
import torch.optim as optim


class TrainPipeline:
    def __init__(
        self,
        config: Namespace,
        yolo: nn.Module,
        optimizer: torch.optim,
        scheduler: optim.lr_scheduler,
        loader: torch.utils.data.DataLoader,
        mode: str = "train",
    ):
        """
        Train the model.

        Parameters
        ----------
            config : argparse.Namespace
                Namespace object, contains all configurations.
            yolo : nn.Module
                Yolov1 model object.
            optimizer : torch.optim
                optimizer object.
            scheduler : torch.optim.lr_scheduler
                - Decreasing learning rate to help reach local minimal.
            loader : DataLoader
                - Dataset loader.
            mode : str
                "train", "test", or "valid" for whether we are training, testing, or validating the model, each will get its corresponding datasets.
        """
        self.config = config
        self.yolo = yolo
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.loader = loader
        self.loss_fn = YoloLoss(config)
        self.mean_loss = None

    def train(self):
        """Train model"""
        print("\n" + "#" * 64)
        print(f"\nTraining Model on {self.mode} dataset.")
        print("\n" + "#" * 64)
        config = self.config

        max_epoch = config.LAST_EPOCH + config.EPOCHS # How many epochs are we taining?

        for epoch in range(config.LAST_EPOCH + 1, config.LAST_EPOCH + config.EPOCHS + 1):
            print("\n\n" + "|" + "-" * 64 + "|")
            print(f"Epoch: {epoch}/{max_epoch} | Lr = {self.optimizer.param_groups[0]['lr'] }")


            # TODO Compute mean_average_precision, look into the res folder the todo
            # mean_average_prec = mean_average_precision(loader=train_loader, model=model, config=config)
            # print(f"\nTrain mAP: {mean_average_prec}")

            # Check Point optional save model if Mean Average Percision is > than num
            # if mean_average_prec > 0.9:
            #     checkpoint = {
            #         "state_dict" : model.state_dict(),
            #         "optimizer" : optimizer.state_dict()
            #     }
            #     save_checkpoint(checkpoint, filename=config.LOAD_MODEL_FILE)

            # === Helper function.
            self.train_one_epoch()

            # === Update Learning Rate: at the end of every epoch. Note: different learning rates need to be updated in different areas of code; example: OneCycleLR (needs per-batch).
            if isinstance(self.scheduler, torch.optim.lr_scheduler.SequentialLR):
                self.scheduler.step()

        # === Save checkpoint.
        checkpoint = {
            "epoch" : epoch,
            "model": self.yolo.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "mean_loss": self.mean_loss
        }
        # checkpoint = {
        #     "state_dict": self.yolo.state_dict(),
        #     "optimizer": self.optimizer.state_dict(),
        # }
        save_checkpoint(state=checkpoint, epochs=epoch, loss=self.mean_loss, config=config)

    def train_one_epoch(self):
        """ <- Helper function takes for each epoch. ->"""
        config = self.config
        loop = tqdm(self.loader, leave=True)
        # store the loss in a list then get the mean of it.
        mean_loss = []

        for batch_idx, (x, y) in enumerate(loop):
            # move tensors to GPU
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # predict
            out = self.yolo(x)

            # reshape
            b_s = x.size(0)  # Batch size not hardcoded for worst-case.
            out = out.view(b_s, config.S, config.S, config.NUM_NODES_PER_CELL)

            # compute loss
            loss = self.loss_fn(out, y)

            mean_loss.append(loss.item())

            self.optimizer.zero_grad()  #  Clear old gradients from the previous step batch (otherwise they'd accumulate).
            loss.backward()  # Compute gradients of the loss w.r.t. model parameters (via backpropagation) i.e -> Gradient Descent.
            self.optimizer.step()  # Update the model’s parameters using the computed gradients

            # update the progress bar
            loop.set_postfix(loss=loss.item())

        self.mean_loss = sum(mean_loss) / len(mean_loss)
        print(f"Mean loss: {sum(mean_loss)/len(mean_loss)}")
