"""
Intersection Over Union
"""

import torch
from data.utils.bbox_conversion import (
    convert_mid_to_corner,
    convert_abs_to_norm,
    convert_norms_to_abs,
    convert_x_y_rel_cell_to_rel_image,
)


def _convert_to_corners(boxes: torch.Tensor, box_format: str = "corners"):
    """
    Helper function to convert midpoint boxes to corner boxes if needed.
    Args:
        boxes (torch.Tensor): Bounding box tensor.
        box_format (str): "corners" or "midpoint".
    Returns:
        torch.Tensor: Bounding box tensor in corner format.
    """
    if box_format == "midpoint":
        x_mid, y_mid, w, h = boxes.unbind(dim=-1)
        x1, y1, x2, y2 = convert_mid_to_corner(x_mid, y_mid, w, h)
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        return boxes


def IoU_one_to_one_mapping(
    pred_boxes: torch.Tensor,
    label_boxes: torch.Tensor,
    S: int = 7,
    box_format: str = "midpoint",
):
    """
    Calculates Intersection Over Union for all the CORRESPONDING bounding box PAIRS across samples in a batch (one-to-one mapping). This function is designed for LOSS computation. When you pass pred_boxes=(N, 4) and label_boxes=(N, 4), it will calculate IoU between pred_box[0] and label_boxes[0] (one-to-one mapping), pred_boxes[1] and label_boxes[1], and so on...

    NOTE: Arg tensors coordinate format needs to be yolo hybrid mid-points where (x, y) are relative to a cell and (w, h) are relative to the image!

    Args:
        pred_boxes (torch.Tensor): Model predictions. Shape: (N, S, S, 4).
            - Where N is the number of bounding boxes in a batch of samples.
            - In YOLO hybrid midpoint.
            - 4 is [x, y, w, h]
        label_boxes (torch.Tensor): Ground truth. Shape: (N, S, S, 4).
            - In YOLO hybrid midpoint.
        S (int): The grid split size.
        box_format (str): The format of the input boxes ("midpoint" or "corners").
            - YOLOv1's raw outputs are a hybrid format (x, y) are relative to the cell but not (w, h).

    Returns:
        torch.Tensor: IoU scores for each box pair. Shape: (N, S, S, 1).
    """
    N = pred_boxes.shape[0]  # Number of boxes

    # --- 1. Create a grid of cell indices (i=rows, j=cols) ---
    j_indices = torch.arange(S, device=pred_boxes.device).repeat(N, S, 1).unsqueeze(-1)

    i_indices = (
        torch.arange(S, device=pred_boxes.device)
        .repeat(N, S, 1)
        .transpose(1, 2)
        .unsqueeze(-1)
    )

    # --- 2. Convert YOLO's hybrid coords to image-relative normalized corner-points ---
    # The raw outputs (pred_boxes, label_boxes) have x,y relative to the cell.
    # w,h are already relative to the entire image.
    x_rel_cell_pred, y_rel_cell_pred, w_pred, h_pred = pred_boxes.unbind(dim=-1)
    x_rel_cell_label, y_rel_cell_label, w_label, h_label = label_boxes.unbind(dim=-1)

    # Convert x, y from relative-to-cell to relative-to-image
    x_mid_norm_pred, y_mid_norm_pred = convert_x_y_rel_cell_to_rel_image(
        x_rel_cell_pred.unsqueeze(-1),
        y_rel_cell_pred.unsqueeze(-1),
        i_indices,
        j_indices,
        S,
    )
    x_mid_norm_label, y_mid_norm_label = convert_x_y_rel_cell_to_rel_image(
        x_rel_cell_label.unsqueeze(-1),
        y_rel_cell_label.unsqueeze(-1),
        i_indices,
        j_indices,
        S,
    )

    # Convert the full normalized midpoint box (x,y,w,h) to normalized corner-points
    pred_x1, pred_y1, pred_x2, pred_y2 = convert_mid_to_corner(
        x_mid_norm_pred, y_mid_norm_pred, w_pred.unsqueeze(-1), h_pred.unsqueeze(-1)
    )
    label_x1, label_y1, label_x2, label_y2 = convert_mid_to_corner(
        x_mid_norm_label, y_mid_norm_label, w_label.unsqueeze(-1), h_label.unsqueeze(-1)
    )

    # --- 3. Calculate Intersection and Union ---
    #   Get the coordinates of the overlapping box where pred_bboxes and label_bboxes overlap (the intersection).
    inter_x1 = torch.max(pred_x1, label_x1)
    inter_y1 = torch.max(pred_y1, label_y1)
    inter_x2 = torch.min(pred_x2, label_x2)
    inter_y2 = torch.min(pred_y2, label_y2)

    # --- 4: Calculate the intersection Area -> the box where the pred box and label box intersect.
    #    the .clamp() is only for cases when the pred and label bounding boxes don't intersect, if they dont it replaces all negative values with 0.
    intersection_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0) # fmt: skip

    # --- 5: Calculate the Union Area -> the union is the total area covered by both label and pred bounding boxes.
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    label_area = (label_x2 - label_x1) * (label_y2 - label_y1)

    union_area = pred_area + label_area - intersection_area

    # --- 6: Calculate Final IoU scores.
    epsilon = 1e-6  # Add a small epsilon to avoid division by zero.
    iou = intersection_area / (union_area + epsilon)

    return iou


def IoU_one_to_many_mapping(
    single_pred_box: torch.Tensor,
    multiple_label_boxes: torch.Tensor,
    box_format: str = "corners",
):
    """
    Calculates Intersection Over Union between one predicted box and multiple ground truth boxes
    (one-to-many mapping). This function is primarily used for mAP calculation, where we
    have a set of predicted boxes and need to compare each one against all ground truth boxes.

    Args:
        single_pred_box (torch.Tensor): A single predicted box. Shape: (4, ).
            - In corner-points with absolute pixel values.
            - 4 is [x1, y1, x2, y2]
        multiple_label_boxes (torch.Tensor): Multiple ground truth boxes. Shape: (N, 4).
            - In corner-points with absolute pixel values.
            - N is the number of bounding boxes
        box_format (str): The format of the input boxes ("corners" or "midpoint").

    Returns:
        torch.Tensor: IoU values (1, N). Where N is the number of iou scores.
    """
    # Use the helper function to ensure both inputs are in corner format for calculation.
    box1 = _convert_to_corners(single_pred_box.unsqueeze(0), box_format)
    box2 = _convert_to_corners(multiple_label_boxes, box_format)

    # Ensure box1 is (1, 4) for broadcasting
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)

    box1_x1, box1_y1, box1_x2, box1_y2 = box1.unbind(dim=-1)
    box2_x1, box2_y1, box2_x2, box2_y2 = box2.unbind(dim=-1)

    # Calculate intersection coordinates
    x1_intersection = torch.max(box1_x1, box2_x1)
    y1_intersection = torch.max(box1_y1, box2_y1)
    x2_intersection = torch.min(box1_x2, box2_x2)
    y2_intersection = torch.min(box1_y2, box2_y2)

    # --- Calculate intersection area
    intersection_area = (x2_intersection - x1_intersection).clamp(0) * (
        y2_intersection - y1_intersection
    ).clamp(0)

    # --- Calculate area of each box
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # --- Calculate union area using broadcasting
    union_area = box1_area + box2_area - intersection_area + 1e-6

    iou = intersection_area / union_area
    return iou.view(-1)  # returned tensor is always 1D


# Test as module:
# $     python -m utils.IoU
def test():
    torch.set_printoptions(threshold=10000)  # prints full tensors!

    from configs.config_loader import load_config
    from data.utils.setup_transforms import setup_transforms
    from data.voc_dataset import VOCDataset
    from data.utils.bbox_extraction import (
        extract_and_convert_label_bboxes,
        extract_and_convert_pred_bboxes,
    )

    cfg = load_config(
        "config_voc_dataset.yaml", verify_ask_user=False, print_configs=False
    )
    t = setup_transforms(cfg.IMAGE_SIZE)
    d = VOCDataset(
        cfg=cfg,
        which_dataset=cfg.TRAIN_DIR_NAME,
        num_samples=cfg.NUM_TRAIN_SAMPLES,
        transforms=t,
    )

    def test_O_to_M():
        print(f"Testing IoU_one_to_one_mapping")
        # --- Create a mini-batch with two samples ---
        img1, label1 = d.__getitem__(0)
        # img2, label2 = d.__getitem__(1)
        # Stack labels to create a batch of size 2
        # label_batch = torch.stack([label1, label2])
        label_batch = torch.stack([label1])

        # Test with the prediction as the same as the label.
        pred_batch = label_batch.clone()

        # Add a slight deviation to one of the predicted boxes to test IoU
        pred_batch[:, :, :, 20:30] += 0.1

        # Reshape to (N, S, S, 4) for the IoU function
        pred_boxes = pred_batch.reshape(-1, cfg.S, cfg.S, cfg.C + cfg.B * 5)[
            ..., cfg.C + 1 : cfg.C + 5
        ]
        label_boxes = label_batch.reshape(-1, cfg.S, cfg.S, cfg.C + cfg.B * 5)[
            ..., cfg.C + 1 : cfg.C + 5
        ]

        iou_scores = IoU_one_to_one_mapping(
            pred_boxes=pred_boxes, label_boxes=label_boxes, box_format="midpoint"
        )
        print("IoU Scores (one-to-one):")
        print(iou_scores)

    def test_M_to_M():
        # --- Test IoU_one_to_many_mapping ---
        print("\n--- Test IoU_one_to_many_mapping ---")
        # A single predicted box in corner-point absolute format
        pred_box = torch.tensor([100.0, 100.0, 150.0, 150.0])

        # Multiple ground truth boxes in a list
        labels = torch.tensor(
            [
                [100.0, 100.0, 150.0, 150.0],  # Perfect match
                [110.0, 110.0, 160.0, 160.0],  # Some overlap
                [200.0, 200.0, 250.0, 250.0],  # No overlap
            ]
        )

        iou_scores_many = IoU_one_to_many_mapping(
            pred_box, labels, box_format="corners"
        )
        print("IoU Scores (one-to-many):")
        print(iou_scores_many)
        # Expected output: iou values close to [1.0, 0.5, 0.0]

    test_M_to_M() # expected tensor([1.0000, 0.4706, 0.0000]) --> 1=Prefect match
    # test_O_to_M()


if __name__ == "__main__":
    test()
