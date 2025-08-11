import torch
import torch.nn as nn
from configs.config_loader import YOLOConfig
from utils.IoU import IoU_one_to_one_mapping


class YOLOLoss(nn.Module):
    def __init__(self, cfg: YOLOConfig):
        """
        YOLO v1 Loss function.

        Args:
            cfg: project configurations.

        """
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # Summed mean squared error.
        self.cfg = cfg
        # The lambdas penalties from paper.
        self.LAMBDA_COORD = 5  # λ_coord
        self.LAMBDA_NOOBJ = 0.5  # λ_noobj

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        """
        Compute YOLOv1 loss.

        Note:
            Where u see example (n_s, S, S, C) or (64, 7, 7) etc.. in the comments below is the tensor's shape.
        Args:
            pred (tensor): shape (batch_size, S, S, CELL_NODES)
            label (tensor): shape (batch_size, S, S, CELL_NODES)

        Returns:
            Tuple(
                Mean_Loss (float),
                Dict: {Separate losses (float)}
                )
        """
        cfg = self.cfg
        S, B, C = cfg.S, cfg.B, cfg.C
        # Batch size = b_s. Hardcode: get the incoming batch size.
        b_s = pred.size(0)

        #  --- 1: Slice out the data. ---
        #      Layout if C=20, B=2 ↓:
        #               [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, pc1, x1, y1, w1, h1, pc2, x2, y2, w2, h2]

        #   Get class predictions for the label and predictions (index 0-C).
        pred_cls = pred[..., :C]  # (b_s, S, S, C)
        label_cls = label[..., :C]

        pred_box1 = pred[..., C : C + 5]  # [pc₁, x₁, y₁, w₁, h₁] -> (b_s, S, S, 5)
        pred_box2 = pred[..., C + 5 : C + 10]  # [pc₂, x2, y₂, w₂, h₂] -> (b_s, S, S, 5)

        label_box = label[..., C : C + 5] # Get probability scores and boxes from the label (only box-1). # fmt: skip

        # Extract components for IoU calculation
        pred_box1_xywh = pred_box1[..., 1:]  # [x₁, y₁, w₁, h₁] -> (b_s, S, S, 4)
        pred_box2_xywh = pred_box2[..., 1:]  # [x2, y₂, w₂, h₂]
        label_xywh = label_box[..., 1:]  # [x₁, y₁, w₁, h₁]

        # --- 2: Calculate IoUs ---
        #  Result determines the "responsible" detector either bbox1 or bbox2 from prediction (compared to label).
        iou_b1 = IoU_one_to_one_mapping(pred_box1_xywh, label_xywh)  #  (b_s, S, S, 1)
        iou_b2 = IoU_one_to_one_mapping(pred_box2_xywh, label_xywh)
        ious = torch.cat([iou_b1, iou_b2], dim=-1)  # (b_s, S, S, 2)

        # --- 3: Find the "responsible" box ---
        #   Get the "Responsible" box (predictions box1 or box2) that is responsible for detecting the object.
        best_box_idx = ious.argmax(-1, keepdim=True)

        # --- 4: Masks ---
        # In the label, the confidence score (pc) is 1 for cells with objects, 0 otherwise.
        obj_mask = label_box[..., 0:1]  # (b_s, S, S, 1)
        noobj_mask = 1.0 - obj_mask #   flip the mask, now a 0 is where an object exists. # fmt:skip

        # --- 5: Gather the responsible predictor's data.
        # We need to gather the responsible bounding box predictions (x, y, w, h and confidence)
        best_pred_xywh = torch.where(
            best_box_idx == 0, pred_box1_xywh, pred_box2_xywh
        )  # (b_s, S, S, 4)
        best_pred_pc = torch.where(
            best_box_idx == 0, pred_box1[..., 0:1], pred_box2[..., 0:1]
        )  # (b_s, S, S, 1)

        # The paper uses the predicted confidence pc_i for this term.
        # The true confidence (C_i) is the IoU of the predicted box with the ground truth box.
        best_pred_iou = torch.gather(ious, -1, best_box_idx)  # (b_s, S, S, 1)

        # =============================== #
        #       Localization Loss         #
        # =============================== #
        # Localization loss or object loss is only for the "responsible" predictor in cells where an object is exists. This loss penalizes the model for inaccurate bounding box predictions.
        #   obj_mask ensures this loss is only calculated for the predictor 'responsible' for an actual object.

        """
        Note in the paper the localization loss is square rooted, however modern approaches removes the square root as bounding box width (w) and height (h) are notoriously difficult to stabilize.
                The original YOLOv1 paper applies the square root to w and h in the loss to:
                    * Penalize small deviations in small boxes more than in large boxes.
                    * Compensate for large variation in scale.
        """
        coord_loss_xy = self.mse(
            obj_mask * best_pred_xywh[..., :2],
            obj_mask * label_xywh[..., :2],
        )
        coord_loss_wh = self.mse(
            obj_mask * torch.sqrt(best_pred_xywh[..., 2:].clamp(min=1e-6)),
            obj_mask * torch.sqrt(label_xywh[..., 2:]),
        )
        localization_loss = coord_loss_xy + coord_loss_wh

        # =============================== #
        #       Confidence Loss           #
        # =============================== #
        # This loss penalizes the model for being wrong about whether an object exists in a cell.

        # Confidence loss for cells with objects
        conf_obj_loss = self.mse(
            obj_mask * best_pred_pc,  # prediction
            obj_mask
            * best_pred_iou,  # truth is IoU of responsible box with the ground truth
        )

        # Confidence loss for cells without objects
        # The paper penalizes ALL confidence predictions in non-object cells.
        pred_box_pcs = torch.cat([pred_box1[..., 0:1], pred_box2[..., 0:1]], dim=-1)
        noobj_loss = self.mse(
            noobj_mask.expand_as(pred_box_pcs)
            * pred_box_pcs,  
            torch.zeros_like(pred_box_pcs),
        )

        # =============================== #
        #       Classification Loss       #
        # =============================== #
        # Classification loss is only for cells with objects.
        # This loss penalizes the model for misclassifying an object (e.g., calling a dog a cat).

        class_loss = self.mse(
            obj_mask.expand_as(pred_cls) * pred_cls,
            obj_mask.expand_as(label_cls) * label_cls,
        )

        # =============================== #
        #       Final compute             #
        # =============================== #

        loss = (
            # <== Localization Loss ==>         For the 'responsible' box either box1 or box2.
            self.LAMBDA_COORD * localization_loss
            #   we multiply by LAMBDA_COORD to give localization loss more weight.
            # <== Confidence Loss ==>           Where an object *is* present.
            + conf_obj_loss
            # <== Confidence Loss ==>           Where *is* object is present.
            + self.LAMBDA_NOOBJ * noobj_loss
            # <== Classification Loss ==>       What is the object? Indices (0-17), compared `to` true label?
            + class_loss
        ) / b_s  # normalize by batch size

        return (
            loss,
            {
                "coord_loss": localization_loss,
                "conf_obj_loss": conf_obj_loss,
                "noobj_loss": noobj_loss,
                "class_loss": class_loss,
            },
        )


# Test as module:
# $     python -m model.loss
def test():
    cfg = load_config(
        "config_voc_dataset.yaml", verify_ask_user=False, print_configs=False
    )
    t = setup_transforms(cfg.IMAGE_SIZE)
    d = VOCDataset(cfg=cfg, which_dataset=cfg.TRAIN_DIR_NAME, transforms=t)

    img, label = d.__getitem__(0)
    label.to(cfg.DEVICE)
    label = label.unsqueeze(0)

    loss_fn = YOLOLoss(cfg)
    # pred = torch.Tensor(1, cfg.S, cfg.S, cfg.CELL_NODES).to(cfg.DEVICE)

    # Mimic the predication as same as label
    pred = label.clone()

    # Since pred is the same as the label add some deviations to alter the x,y, wtc..
    # pred[..., cfg.C + 1] += 0.1  # Slightly shift x-coordinate of bounding box
    # pred[..., 0] = 1  # Set class 0 confidence instead of correct one
    # pred[..., 0:17] = 2  # set all classes predictions to 2

    loss = loss_fn(pred, label)
    print("\nFINAL LOSS:", loss)


if __name__ == "__main__":
    from configs.config_loader import load_config
    from data.voc_dataset import VOCDataset

    torch.set_printoptions(
        threshold=torch.inf
    )  # shows all the values when printing tensors
    from data.utils.setup_transforms import setup_transforms

    test()
