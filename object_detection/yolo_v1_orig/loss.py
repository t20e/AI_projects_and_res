import torch
import torch.nn as nn
from configs.config_loader import YOLOConfig
from utils.IoU import IoU


class YOLOLoss(nn.Module):
    def __init__(self, cfg: YOLOConfig):
        """
        YOLO v1 Loss function.

        Args:
            cfg: project configurations.

        """
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(
            reduction="sum"
        )  #  CrossEntropy/Softmax losses normally work better than MSE.
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
            Loss (float)
        """
        cfg = self.cfg
        S, B, C = cfg.S, cfg.B, cfg.C
        # Batch size = b_s. Get the incoming batch size.
        b_s = pred.size(0)

        # ==> 1: Slice out the data.
        #      Layout if C=20, B=2 ↓:
        #               [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, pc1, x1,y1,w1,h1, pc2, x2,y2,w2,h2]

        #   Get class predictions for the label and predictions (index 0-C).
        pred_cls = pred[..., :C]  # (b_s, S, S, C)
        label_cls = label[..., :C]
        #   Get probability scores and boxes from the predictions.
        pred_box1_pc = pred[..., C : C + 1]  # (b_s, S, S, 1)
        pred_box1_xywh = pred[..., C + 1 : C + 5]  # (b_s, S, S, 4)

        pred_box2_pc = pred[..., C + 5 : C + 6]  # (b_s, S, S, 1)
        pred_box2_xywh = pred[..., C + 6 : C + 10]  # (b_s, S, S, 4)

        #   Get probability scores and boxes from the label (only box-1).
        label_pc = label[..., C : C + 1]  # (b_s, S, S, 1)
        label_xywh = label[..., C + 1 : C + 5]  # (b_s, S, S, 4)

        # ==> Calculate IoUs to determine the "responsible" detector either bbox1 or bbox2 from pred (compared to label).
        iou_b1 = IoU(pred_box1_xywh, label_xywh)  # (b_s, S, S, 1)
        iou_b2 = IoU(pred_box2_xywh, label_xywh)
        ious = torch.cat([iou_b1, iou_b2], dim=-1)  # (b_s, S, S, 2)

        #   Get the "Responsible" box (predictions box1 or box2) that is responsible for detecting the object.
        best_box_idx = ious.argmax(-1, keepdim=True)

        #   Create a Mask
        obj_mask = label_pc  # The mask will have a value of 1 where an object exists.
        #   flip the mask, now a 0 is where an object exists.
        noobj_mask = 1.0 - obj_mask

        # ==> 4: Gather the "responsible" detector with its pc and xywh.
        best_xywh = torch.where(
            best_box_idx == 0, pred_box1_xywh, pred_box2_xywh
        )  # (b_s, S, S, 4)
        best_pc = torch.where(
            best_box_idx == 0, pred_box1_pc, pred_box2_pc
        )  # (b_s, S, S, 1)
        best_iou = torch.gather(ious, -1, best_box_idx)  # (b_s, S, S, 1)

        # =============================== #
        #       Localization Loss         #
        # =============================== #
        # Localization loss or object loss is only for the "responsible" predictor in cells where an object is exists. This loss penalizes the model for inaccurate bounding box predictions.
        coord_loss = (
            self.mse(  # first formula of the loss function in paper illustration.
                obj_mask * best_xywh[..., :2],
                obj_mask * label_xywh[..., :2],
            )
        )
        #   obj_mask ensures this loss is only calculated for the predictor 'responsible' for an actual object.
        coord_loss += (
            self.mse(  # Second formula of the loss function in paper illustration.
                obj_mask * torch.sqrt(best_xywh[..., 2:4].clamp(min=1e-6)),
                obj_mask * torch.sqrt(label_xywh[..., 2:4]),
            )
        )

        # =============================== #
        #       Confidence Loss           #
        # =============================== #
        # This loss penalizes the model for being wrong about whether an object exists in a cell.
        conf_obj_loss = self.mse(obj_mask * best_pc, obj_mask * best_iou)

        best_pred_pcs = torch.cat([pred_box1_pc, pred_box2_pc], dim=-1)  # (64, 7, 7, 2)
        noobj_loss = self.mse(
            noobj_mask * best_pred_pcs, torch.zeros_like(best_pred_pcs)
        )

        # =============================== #
        #       Classification Loss       #
        # =============================== #
        # This loss penalizes the model for misclassifying an object (e.g., calling a dog a cat).
        class_loss = self.mse(obj_mask * pred_cls, obj_mask * label_cls)

        # =============================== #
        #       Final compute             #
        # =============================== #

        loss = (
            # <== Localization Loss ==>         For the 'responsible' box either box1 or box2.
            self.LAMBDA_COORD * coord_loss
            #   we multiply by LAMBDA_COORD to give localization loss more weight.
            # <== Confidence Loss ==>           Where an object *is* present.
            + conf_obj_loss
            # <== Confidence Loss ==>           Where *is* object is present.
            + self.LAMBDA_NOOBJ * noobj_loss
            # <== Classification Loss ==>       What is the object? Indices (0-17), compared `to` true label?
            + class_loss
        ) / b_s  # normalize by batch.

        # --- Debug print statement to show separate losses values.
        # print(
        #     f"\n\ncoord_loss: {coord_loss}, \nconf_obj_loss: {conf_obj_loss}, \nnoobj_loss: {noobj_loss}, \nclass_loss: {class_loss}\n\n"
        # )

        return loss


# Test as module:
# $ python -m loss
def test(label):
    loss_fn = YOLOLoss(cfg)
    # Create test tensors note the pred tensors is random
    pred = torch.Tensor(1, cfg.S, cfg.S, cfg.CELL_NODES).to(cfg.DEVICE)
    # add one batch dimension to label
    label = label.unsqueeze(0)
    loss = loss_fn(pred, label)
    print("\nFINAL LOSS:", loss)


if __name__ == "__main__":
    from configs.config_loader import load_config
    from data.voc_dataset import VOCDataset

    torch.set_printoptions(
        threshold=torch.inf
    )  # shows all the values when printing tensors
    from data.utils.setup_transforms import setup_transforms

    cfg = load_config("yolov1.yaml")

    t = setup_transforms(cfg.IMAGE_SIZE)
    d = VOCDataset(cfg=cfg, transforms=t)
    img, label = d.__getitem__(0)
    test(label.to(cfg.DEVICE))
