import torch
import torch.nn as nn
from utils.intersection_over_union import intersection_over_union
import math


class YoloLoss(nn.Module):
    """
    Yolo Loss implementation.
    
        When training the model, we will evaluate the models accuracy using this loss function.
    
    Parameters
    ----------
        config : argparse.Namespace
            configurations
    
    """
    def __init__(self, config):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.config = config
        self.lambda_noobj = 0.5 # no object
        self.lambda_coord = 5 # constant
        
    def forward(self, predictions, target): # call function
        """
            Predictions: model predictions
            Target: true labels
        """
        config = self.config
        S = config.S
        B = config.B
        C = config.C
        
        
        predictions = predictions.reshape(-1, S, S, C + B * 5)
        # NOTE: ERROR HERE NUM_NODES_PER_CELL =  1,372 not 28
        
        # [..., 4:config.NUM_NODES_PER_CELL-5] grabs from 4:8 first bbox x,y,w,h
        # [..., 9:config.NUM_NODES_PER_CELL] grabs from 9:13 second bbox x,y,w,h
        
        iou_bbox1 = intersection_over_union(predictions[..., 4:config.NUM_NODES_PER_CELL-5], target[..., 4:config.NUM_NODES_PER_CELL-5])
        # 0 to 2 indexs is going to be for class probabilities
        # 3 is going to be for the class score
        # 4 to 8 is for x, w, y, h for the first bbox in a cell.
        
        iou_bbox2 = intersection_over_union(predictions[..., 9:config.NUM_NODES_PER_CELL], target[..., 4:config.NUM_NODES_PER_CELL-5]) # target stays the same because we are comparing the same cell
        
        # we unsqueeze to add a dimension at the beginning so that we can concatenate them along the first dimension
        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim=0)
        
        # iou_maxes will return the max iou value for each bounding box
        # bestbox will return the argmax of the best bounding box
        iou_maxes, bestbox = torch.max(ious, dim=0)
        
        # at index 3 its going to be 0 or 1 depending on if theres an object in that cell
        exists_box = target[..., 3].unsqueeze(3) # identity of obj i in paper, which tells us is there an object in cell i
            
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_predictions = exists_box * (
            (
            # Check which is the best box, it will be a 1 if the second bbox is best or 0 if the first is best.
                bestbox * predictions[..., 9:config.NUM_NODES_PER_CELL]
                # however if the other one was best then
                + (1 - bestbox) * predictions[..., 4:config.NUM_NODES_PER_CELL-5] # Pc1, X, Y, W, H first bounding box
            )
        )
        
        box_targets = exists_box * target[..., 4:config.NUM_NODES_PER_CELL-5]
        
        # Take the square root of the w/width, h/height of the bounding box 
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6) # + 1e-6 # 1e-6 is to prevent division by zero
        )
            
            
        # these are the labels so they wont be negative
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        
        # ======================== #
        #   FOR OBJECT LOSS        #
        # ======================== #
        # pred_box is the confidence score for the best bounding box
        pred_box = (
            bestbox * predictions[..., 3:4] + (1 - bestbox) * predictions[..., 8:9]
        )
        
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 8:9])
        )
        
        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #
        # for bounding box 1
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 3:4], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 3:4], start_dim=1)
        )
        # for bounding box 2
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 8:9], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 3:4], start_dim=1)
        )
        
        
        # ======================== #
        #   FOR CLASS LOSS         #
        # ======================== #
        
        # (N,S,S,3) -> (N*S*S, 3) when we use end_dim=-2
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :3], end_dim=-2),
            torch.flatten(exists_box * target[..., :3], end_dim=-2)
        )
        
        loss = (
            self.lambda_coord * box_loss # first two rows of functions in the loss in orginal V1 paper, which are for the bounding boxes
            
            + object_loss # third row loss function, if theres an object in the cell
            + self.lambda_noobj * no_object_loss # fouth row loss function, if theres no object in that cell
            + class_loss # fifth row loss function, the class loss
        )
        
        return loss