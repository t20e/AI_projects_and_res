"""
Intersection Over Union
"""

import torch

def IoU_one_to_one_mapping(
    pred_boxes: torch.Tensor,
    label_boxes: torch.Tensor,
    box_format: str = "midpoint",
):
    """
    Calculates Intersection Over Union for CORRESPONDING bounding box PAIRS in a batch, one-to-one mapping.

    When you pass pred_boxes=(batch_size, 4) and label_boxes=(batch_size, 4), it will calculate IoU between pred_box[0] and label_boxes[0] (one-to-one mapping), pred_boxes[1] and label_boxes[1], and so on.

    Args:
        pred_boxes: Model bounding box predictions. Shape: (batch_size, 4).
        label_boxes: Labeled bounding boxes. Shape: (batch_size, 4).
        box_format (str): What format are the passed in bounding box tensors in, "midpoint" or "corners".
            - "midpoint": [x, y, w, h]
            - "corners": [x1, y1, x2, y2] or [x_min, y_min, x_max, y_max]
    Returns:
        tensor: Intersection Over Union for all examples in a batch. Shape (N, 1)

    """
    # --- 1: If "midpoint" convert to corner-points, its easier to calculate IoU with corners.
    if box_format == "midpoint":
        # [..., 0:1] is the midpoint, [..., 2:3] is the width of that bbox, so if we divide it by 2, then well get the top-left x1 coordinate
        pred_x1 = pred_boxes[..., 0:1] - pred_boxes[..., 2:3] / 2
        pred_y1 = pred_boxes[..., 1:2] - pred_boxes[..., 3:4] / 2
        pred_x2 = pred_boxes[..., 0:1] + pred_boxes[..., 2:3] / 2
        pred_y2 = pred_boxes[..., 1:2] + pred_boxes[..., 3:4] / 2
        label_x1 = label_boxes[..., 0:1] - label_boxes[..., 2:3] / 2
        label_y1 = label_boxes[..., 1:2] - label_boxes[..., 3:4] / 2
        label_x2 = label_boxes[..., 0:1] + label_boxes[..., 2:3] / 2
        label_y2 = label_boxes[..., 1:2] + label_boxes[..., 3:4] / 2
    # --- 2: Grab the corner points
    elif box_format == "corners":
        pred_x1 = pred_boxes[..., 0:1]  # the ... slices but keeps the same dimensions.
        pred_y1 = pred_boxes[..., 1:2]
        pred_x2 = pred_boxes[..., 2:3]
        pred_y2 = pred_boxes[..., 3:4]
        label_x1 = label_boxes[..., 0:1]
        label_y1 = label_boxes[..., 1:2]
        label_x2 = label_boxes[..., 2:3]
        label_y2 = label_boxes[..., 3:4]

    # --- 3: Calculate the intersection -> get the coordinates of the overlapping box where pred_bboxes and label_bboxes overlap (the intersection).
    inter_x1 = torch.max(pred_x1, label_x1)
    inter_y1 = torch.max(pred_y1, label_y1)
    inter_x2 = torch.min(pred_x2, label_x2)
    inter_y2 = torch.min(pred_y2, label_y2)

    # --- 4: Calculate the intersection Area -> the box where the pred box and label box intersect.
    #    the .clamp() is only for cases when the pred and label bounding boxes don't intersect, if they dont it replaces all negative values with 0.
    intersection_area = (inter_x2 - inter_x1).clamp(min=0) * (
        inter_y2 - inter_y1
    ).clamp(min=0)

    # --- 5: Calculate the Union Area -> the union is the total area covered by both label and pred bounding boxes.
    pred_area = abs(pred_x2 - pred_x1) * abs(pred_y2 - pred_y1)
    label_area = abs(label_x2 - label_x1) * abs(label_y2 - label_y1)

    union_area = pred_area + label_area - intersection_area

    # --- 6: Calculate Final IoU scores.
    epsilon = 1e-6  # Add a small epsilon to avoid division by zero.
    return intersection_area / (union_area + epsilon)


def IoU_one_to_many_mapping(
    single_pred_box: torch.Tensor,
    multiple_label_boxes: torch.Tensor,
    box_format: str = "corners",
):
    """
    Calculates Intersection Over Union between one predicted box and multiple ground truth boxes (one-to-many mapping).

    Args:
        single_pred_box (torch.Tensor): A single predicted box. Shape: (1, 4).
        multiple_label_boxes (torch.Tensor): Multiple ground truth boxes. Shape: (N, 4).
        box_format (str): "corners" for [x1, y1, x2, y2] or "midpoint" for [x_center, y_center, width, height]
    Returns:
        torch.Tensor: IoU values (1, N). Where N is the number of iou scores.
    """
    # Ensure inputs are correctly shaped for broadcasting
    # single_pred_box should be (1, 4) or (4,) which will be unsqueezed to (1,4) by below
    # multiple_label_boxes should be (M, 4)

    # Convert to corners for calculation
    if box_format == "midpoint":
        box1_x1 = single_pred_box[..., 0:1] - single_pred_box[..., 2:3] / 2
        box1_y1 = single_pred_box[..., 1:2] - single_pred_box[..., 3:4] / 2
        box1_x2 = single_pred_box[..., 0:1] + single_pred_box[..., 2:3] / 2
        box1_y2 = single_pred_box[..., 1:2] + single_pred_box[..., 3:4] / 2

        box2_x1 = multiple_label_boxes[..., 0:1] - multiple_label_boxes[..., 2:3] / 2
        box2_y1 = multiple_label_boxes[..., 1:2] - multiple_label_boxes[..., 3:4] / 2
        box2_x2 = multiple_label_boxes[..., 0:1] + multiple_label_boxes[..., 2:3] / 2
        box2_y2 = multiple_label_boxes[..., 1:2] + multiple_label_boxes[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = single_pred_box[..., 0:1]
        box1_y1 = single_pred_box[..., 1:2]
        box1_x2 = single_pred_box[..., 2:3]
        box1_y2 = single_pred_box[..., 3:4]

        box2_x1 = multiple_label_boxes[..., 0:1]
        box2_y1 = multiple_label_boxes[..., 1:2]
        box2_x2 = multiple_label_boxes[..., 2:3]
        box2_y2 = multiple_label_boxes[..., 3:4]

    # Key for broadcasting:
    # box1_x1 will be (1, 1) if single_pred_box is (1, 4)
    # box2_x1 will be (M, 1)
    # Transposing box2_x1 to (1, M) allows element-wise max/min with box1_x1
    x1_intersection = torch.max(box1_x1, box2_x1.transpose(-1, -2))
    y1_intersection = torch.max(box1_y1, box2_y1.transpose(-1, -2))
    x2_intersection = torch.min(box1_x2, box2_x2.transpose(-1, -2))
    y2_intersection = torch.min(box1_y2, box2_y2.transpose(-1, -2))

    intersection_area = (x2_intersection - x1_intersection).clamp(0) * (
        y2_intersection - y1_intersection
    ).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # union_area = (1, 1) + (1, M) - (1, M) -> (1, M)
    union_area = box1_area + box2_area.transpose(-1, -2) - intersection_area + 1e-6

    iou = intersection_area / union_area

    return iou


# Test as module:
# $     python -m utils.IoU
def test():
    from configs.config_loader import load_config

    cfg = load_config("config_voc_dataset.yaml")
    # Create test tensors note these will not match the actual pred/label tensors
    pred = torch.Tensor(cfg.BATCH_SIZE, cfg.S, cfg.S, 4).to(cfg.DEVICE)
    label = torch.Tensor(cfg.BATCH_SIZE, cfg.S, cfg.S, 4).to(cfg.DEVICE)

    i = IoU_one_to_one_mapping(pred_boxes=pred, label_boxes=label, box_format="midpoint")
    print(i)


if __name__ == "__main__":
    test()
