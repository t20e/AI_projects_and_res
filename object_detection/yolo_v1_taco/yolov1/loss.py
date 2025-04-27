import torch
import torch.nn as nn
from argparse import Namespace

from utils.intersection_over_union import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, config: Namespace):
        """
        Yolo Loss implementation.

            When training the model, we will evaluate the models accuracy using the loss function from the paper.

        Parameters
        ----------
            config : argparse.Namespace
                configurations

        """
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.config = config
        self.lambda_noobj = 0.5  # no object exists.
        self.lambda_coord = 5  # constant from the paper.

    def forward(self, predictions, target):  # call function
        """
        Predictions: model predictions
        Target: true labels
        """
        config = self.config
        S = config.S
        B = config.B
        C = config.C

        # -> reshapes into (BATCH_SIZE, S, S, NUM_NODES_PER_CELL)
        predictions = predictions.reshape(-1, S, S, C + B * 5)

        # =============================== #
        #   SET UP BBOX COORDINATE SLICES #
        # =============================== #

        #   a cell tensor = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
        #   -> first bbox x, y, w, h = [19, 20, 21, 22]
        #   -> second bbox x, y, w, h = [24, 25, 26, 27]
        #   -> indexes 18 and 23 store pc_1 and pc_2 scores.

        first_bbox_coord_slice = slice( # grab the x, y, w, h
            config.C + 1, config.C+5
        )
        second_bbox_coord_slice = slice( # grab the x, y, w, h
            config.NUM_NODES_PER_CELL - 4, config.NUM_NODES_PER_CELL
        )

        # Set up slices to grab confidence scores/pc_1 and pc_2
        pc_1_slice = slice(config.C, config.C + 1)  # grab index 18
        pc_2_slice = slice(config.C + 5, config.C + 5 + 1)  # grab index 23

        iou_bbox1 = intersection_over_union(
            predictions[..., first_bbox_coord_slice],
            target[..., first_bbox_coord_slice],
        )

        iou_bbox2 = intersection_over_union(
            # target slice stays the same because the label/true only has one bounding box
            predictions[..., second_bbox_coord_slice],
            target[..., first_bbox_coord_slice],
        )

        # we unsqueeze to add a dimension at the beginning so that we can concatenate them along the first dimension
        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim=0)

        # iou_maxes will return the max iou value for each bounding box
        # bestbox will return the argmax of the best bounding box
        iou_maxes, bestbox = torch.max(ious, dim=0)

        # at index 18 its going to be 0 or 1 depending on if theres an object in that cell
        # identity of obj i in paper, which tells us is there an object in cell i
        exists_box = target[..., config.C].unsqueeze(
            3  # 3 reshapes from (1, 7, 7) -> to -> (1, 7, 7, 1)
        )

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_predictions = exists_box * (
            (
                # Check which is the best box, it will be a 1 if the second bbox is best or 0 if the first is best.
                bestbox * predictions[..., second_bbox_coord_slice]
                # however if the other one was best then
                + (1 - bestbox) * predictions[..., first_bbox_coord_slice]
            )
        )

        box_targets = exists_box * target[..., first_bbox_coord_slice]

        # Take the square root of the w/width, h/height of the bounding box
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(
                box_predictions[..., 2:4] + 1e-6
            )  # + 1e-6 # 1e-6 is to prevent division by zero
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
            bestbox * predictions[..., pc_2_slice] + (1 - bestbox) * predictions[..., pc_1_slice]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., pc_1_slice]),
        )

        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #
        # for bounding box 1
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., pc_1_slice], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., pc_1_slice], start_dim=1),
        )
        # for bounding box 2
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., pc_2_slice], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., pc_1_slice], start_dim=1),
        )

        # ======================== #
        #   FOR CLASS LOSS         #
        # ======================== #

        # (N,S,S,3) -> (N*S*S, 3) when we use end_dim=-2
        class_loss = self.mse(
            # [..., :config.C] gets all the class values -> 0 to 17s
            torch.flatten(exists_box * predictions[..., :config.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :config.C], end_dim=-2),
        )

        loss = (
            self.lambda_coord
            * box_loss  # first two rows of functions in the loss in orginal V1 paper, which are for the bounding boxes
            + object_loss  # third row loss function, if theres an object in the cell
            + self.lambda_noobj
            * no_object_loss  # fouth row loss function, if theres no object in that cell
            + class_loss  # fifth row loss function, the class loss
        )

        return loss
