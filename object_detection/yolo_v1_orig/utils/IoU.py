"""
Intersection Over Union
"""

import torch


def IoU(
    pred_boxes: torch.Tensor,
    label_boxes: torch.Tensor,
    box_format: str = "midpoint",
):
    """
    Calculates Intersection Over Union on a batch of predicted bounding boxes compared the the target labeled boxes.

    Args:
        pred_boxes: Model bounding box predictions. Shape: (batch_size, 4).
        label_boxes: Labeled bounding boxes. Shape: (batch_size, 4).
        box_format (str): What format are the passed in bounding box tensors in, "midpoint" or "corners".
            - "midpoint": [x, y, w, h]
            - "corners": [x1, y1, x2, y2] or [x_min, y_min, x_max, y_max]
    Returns:
        tensor: Intersection Over Union for all examples in a batch.
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


# Test as module:
# $ python -m utils.IoU
def test():
    from configs.config_loader import load_config

    cfg = load_config("yolov1.yaml")
    # Create test tensors note these will not match the actual pred/label tensors
    pred = torch.Tensor(cfg.BATCH_SIZE, cfg.S, cfg.S, 4).to(cfg.DEVICE)
    label = torch.Tensor(cfg.BATCH_SIZE, cfg.S, cfg.S, 4).to(cfg.DEVICE)

    i = IoU(pred_boxes=pred, label_boxes=label, box_format="midpoint")
    print(i)


if __name__ == "__main__":
    test()
