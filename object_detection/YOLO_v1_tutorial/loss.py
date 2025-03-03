import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        # 0 to 19 is going to be for class probabilities
        # 20 is going to be for the class score
        # 21 to 25 is for the four bounding boxes for the first cell.
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) #target size stays the same because we are comparing the same cell
        
        # we unsqueeze to add a dimension at the beginning so that we can concatenate them along the first dimension
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # iou_maxes will return the max iou value for each bounding box
        # bestbox will returnt he argmax of the best bounding box
        iou_maxes, bestbox = torch.max(ious, dim=0)
        
        # at 20 its going to be 0 or 1 depending on if theres an object in that cell
        exists_box = target[..., 20].unsqueeze(3) # identity of obj i in paper, which tells us is there an object in cell i
        
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_predictions = exists_box * (
            (
            # check which is the best box, this is going to be 1 if the second bounding box is best second bounding box is from 26 to 30 Pc2, X, Y, W, H
                bestbox * predictions[..., 26:30] 
                # however if the other one was best then
                + (1 - bestbox) * predictions[..., 21:25] # Pc1, X, Y, W, H first bounding box
            )
        )
        
        box_targets = exists_box * target[..., 21:25]
        
        # Take the square root of the width and height of the bounding box 
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6) # + 1e-6 # 1e-6 is to prevent division by zero
        )
            
            
        # these are labels so they wont be negative
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
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        
        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #
        # for bounding box 1
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        # for bounding box 2
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        
        
        # ======================== #
        #   FOR CLASS LOSS         #
        # ======================== #
        
        # (N,S,S,20) -> (N*S*S, 20) when we use end_dim=-2
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        loss = (
            self.lambda_coord * box_loss # first two rows of functions in the loss in orginal V1 paper, which are for the bounding boxes
            
            + object_loss # third row loss function, if theres an object in the cell
            + self.lambda_noobj * no_object_loss # fouth row loss function, if theres no object in that cell
            + class_loss # fifth row loss function, the class loss
        )
        
        return loss