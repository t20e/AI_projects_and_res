import torch
import torch.nn as nn
from argparse import Namespace

from utils.intersection_over_union import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, config: Namespace):
        """
        Yolo v1 Loss function.

        Note: While extracting bboxes (extract_bboxes()) seems intuitive, its better to keep the tensor in its grid structure of -> (batch_size, S, S, 28). 28 is number of nodes per cell.

        Args:
            config (argparse.Namespace): project configurations.

        """
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(
            reduction="sum"
        )  #  CrossEntropy/Softmax losses normally work better than MSE.
        self.config = config
        # λ_coord & λ_noobj from paper.
        self.LAMBDA_COORD = 5
        self.LAMBDA_NOOBJ = 0.5

    def forward(self, pred: torch.Tensor, label: torch.Tensor):  # call function
        """
        Compute loss.

        Note:
            Where u see example (n_s, S, S, C) or (64, 7, 7) etc.. in the comments below are the tensor's shape.
        Args:
            pred (tensor) : shape (batch_size, S, S, num_nodes_per_cell)
            label (tensor) : shape (batch_size, S, S, num_nodes_per_cell)

        Returns:
            Loss (float)

        """
        S, B, C, DEVICE, NUM_NODES_PER_CELL = (
            self.config.S,  # S=7 for example shape comments below
            self.config.B,  # B=2                 ↓
            self.config.C,  # C=18                ↓
            self.config.DEVICE,
            self.config.NUM_NODES_PER_CELL,  # =28
        )

        # Just in case for b_s: Use the actual batch size coming through the network (pred.size(0)). E.g. BATCH_SIZE = 64 with 132 samples and u choose not to discard batches with examples less than BATCH_SIZE -> you get two full batches of 64 samples and one final batch of 4 samples. When we later ( / b_s ) it will cause issues. Instead normalize by the size of the batch, in this example batch:64 / 64, and batch with 4 sample/4.
        b_s = pred.size(0)


        # ==> 1: Slice out the coordinates we need.
        #       Layout: C=18, B=2.
        #            [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, pc1, x1,y1,w1,h1, pc2, x2,y2,w2,h2]
        #   Get class predictions index 0-17. (n_s, S, S, C)
        pred_cls = pred[..., :C]

        # TODO added one line below understand Y, ALSO: The original YOLO v1 paper used a sum-squared error loss for all predictions, including class probabilities. It did not apply softmax to the class predictions before calculating the MSE loss. 
        # pred_cls = torch.softmax(pred_cls, dim=-1)  



        label_cls = label[..., :C]

        #   Get bbox1 pc, and x,y,w,h
        pred_box1_pc = pred[
            ..., C : C + 1
        ]  # Get bbox1 probability score index(18) pc_1. Shape: (64, 7, 7, 1)
        pred_box1_xywh = pred[
            ..., C + 1 : C + 5
        ]  # Get bbox1 x,y,w,h coordinates indexes(19, 20, 21, 22). Shape: (64, 7, 7, 4)

        #   Get bbox2 pc, and x,y,w,h
        pred_box2_pc = pred[
            ..., C + 5 : C + 6
        ]  # Get bbox1 probability score @ index(23) pc_2. Shape: (64, 7, 7, 1)
        pred_box2_xywh = pred[
            ..., C + 6 : C + 10
        ]  # Get bbox2 x,y,w,h coordinates @ indexes(24, 25, 26, 27). Shape: (64, 7, 7, 4)

        # Get label data, only bbox1 is filled for labeled data, pc (index) which the object=1 and so is the class prediction index (0-17).
        label_pc = label[..., C : C + 1]  # Shape: (64, 7, 7, 1)
        label_xywh = label[..., C + 1 : C + 5]  # Shape: (64, 7, 7, 4)

        # ==> 2: Calculate IoUs to determine the "responsible" detector either bbox1 or bbox2 in every cell.
        iou_b1 = intersection_over_union(pred_box1_xywh, label_xywh)  # (64, 7, 7, 1)
        iou_b2 = intersection_over_union(pred_box2_xywh, label_xywh)  # (64, 7, 7, 1)

        ious = torch.cat([iou_b1, iou_b2], dim=-1)  # (64, 7, 7, 2)
        best_idx = ious.argmax(
            -1, keepdim=True
        )  # (64, 7, 7, 1). Get the best bounding boxes index. (0 -> bbox1, 1 -> bbox2)

        # ==> 3: Create a mask.
        obj_mask = (
            label_pc  # (64, 7, 7, 1) Give it a value of 1 where an object exists.
        )
        noobj_mask = 1.0 - obj_mask  # flip the mask, now a 0 is where an object exists.

        # ==> 4: Gather "responsible" detector with its pc and bbox.
        best_xywh = torch.where(
            best_idx == 0, pred_box1_xywh, pred_box2_xywh
        )  # (64, 7, 7, 4)
        best_pc = torch.where(
            best_idx == 0, pred_box1_pc, pred_box2_pc
        )  # (64, 7, 7, 1)
        best_iou = torch.gather(ious, -1, best_idx)  # (64, 7, 7, 1)

        # ==>============================ #
        #       Localization Loss         #
        # ==>============================ #
        # This loss is only for the "responsible" predictor in cells where an object is exists.
        coord_loss = self.mse(  
            obj_mask * best_xywh[..., :2],
            obj_mask * label_xywh[..., :2],
        )
        coord_loss += self.mse(  
            obj_mask * torch.sqrt(best_xywh[..., 2:4].clamp(0)),
            obj_mask * torch.sqrt(label_xywh[..., 2:4]),
        )

        # ==>============================ #
        #       Confidence Loss           #
        # ==>============================ #
        conf_obj_loss = self.mse(  
            obj_mask * best_pc, obj_mask * best_iou
        )

        best_pred_pcs = torch.cat([pred_box1_pc, pred_box2_pc], dim=-1)  # (64, 7, 7, 2)
        noobj_loss = self.mse(  
            noobj_mask * best_pred_pcs, noobj_mask * torch.zeros_like(best_pred_pcs)
        )

        # ==>============================ #
        #       Classification Loss       #
        # ==>============================ #
        class_loss = self.mse( 
            obj_mask * pred_cls, obj_mask * label_cls
        )

        # ==>============================ #
        #       Final compute             #
        # ==>============================ #

        print(  # Test print statement to show separate loss values.
            f"\n\ncoord_loss: {coord_loss}, \nconf_obj_loss: {conf_obj_loss}, \nnoobj_loss: {noobj_loss}, \nclass_loss: {class_loss}\n\n"
        )

        loss = (  
            # <== Localization Loss ==>         For the 'responsible' box either box1 or box2.
            self.LAMBDA_COORD * coord_loss
            # <== Confidence Loss ==>           Where an object *is* present.
            + conf_obj_loss
            # <== Confidence Loss ==>           Where *is* object is present.
            + self.LAMBDA_NOOBJ * noobj_loss
            # <== Classification Loss ==>       What is the object? Indices (0-17), compared `to` true label?
            + class_loss
        ) / b_s  # normalize by batch.

        return loss
