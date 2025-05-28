import torch 
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import Namespace
import torch.nn as nn
import torch.optim
import torchvision.transforms as T

from .loss import YoloLoss
from .dataset import Dataset 

from utils.checkpoints import save_checkpoint


# from utils.mean_average_precision import mean_average_precision
        


class Train():
    def __init__(self, config:Namespace, yolo:nn.Module, optimizer:torch.optim, transforms:T, mode:str="train"):
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
            transforms : torchvision.transforms
                transform object to resize the bboxes and images.  Normalize image tensors.
            mode : str
                "train", "test", or "valid" for whether we are training, testing, or validating the model, each will get its corresponding datasets.
        """
        self.config = config
        self.yolo = yolo
        self.optimizer = optimizer
        self.transforms = transforms
        self.mode = mode
        self.loss_fn = YoloLoss(config)
        print(f"\n\nTraining model with the {mode} dataset!\n\n")
        self.mean_loss = None
        self.setup_loader()
        self.train()

    def setup_loader(self):
        """Set up the dataset loader"""
        config = self.config

        dataset = Dataset(S=config.S, B=config.B, C=config.C, mode=self.mode, dataset_path=config.DATASET_DIR, transforms=self.transforms)
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=True
        )

    def train(self):
        """Train model."""
        config = self.config

        for epoch in range(config.EPOCHS):
            print("\n\n" + "|" + "-" * 64 + "|")
            print("Epoch:", epoch + 1)
            
            # Compute mean_average_percision # TODO
            # mean_average_prec = mean_average_precision(loader=train_loader, model=model, config=config)
            # print(f"\nTrain mAP: {mean_average_prec}")

            # Check Point optional save model if Mean Average Percision is > than num
            # if mean_average_prec > 0.9:
            #     checkpoint = {
            #         "state_dict" : model.state_dict(),
            #         "optimizer" : optimizer.state_dict()
            #     }
            #     save_checkpoint(checkpoint, filename=config.LOAD_MODEL_FILE)
        
            self.run_gradient_descent()
        
        # Save checkpoint after training the model for a certain number of epochs
        checkpoint = {
            "state_dict" : self.yolo.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        save_checkpoint(checkpoint, config.EPOCHS, self.mean_loss, self.config)

    def run_gradient_descent(self):
        """Run gradient descent to train the model"""
        config = self.config
        loop = tqdm(self.loader, leave=True)
        # store the loss in a list then get the mean of it.
        mean_loss = []
        
        
        for batch_idx, (x, y) in enumerate(loop):
            # move tensors to GPU
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            # predict 
            out = self.yolo(x)
            # compute loss
            loss = self.loss_fn(out, y)
            
            mean_loss.append(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # TODO implemenet a plot predictions while training the model, but do it at random epochs somewhere closer to the end of training.
            # if config.PLOT_WHILE_TRAINING:
            #     pass
            #update the progress bar
            loop.set_postfix(loss=loss.item())
        
        self.mean_loss = sum(mean_loss)/len(mean_loss)
        print(f"Mean loss: {sum(mean_loss)/len(mean_loss)}")