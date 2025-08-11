"""
Utils to convert coordinates from mid-point to corner-points and vice versa.
"""

import torch

# My modules
from configs.config_loader import load_config, YOLOConfig
from data.voc_dataset import VOCDataset
from data.utils.setup_transforms import setup_transforms
from data.utils.bbox_conversion import (
    convert_mid_to_corner,
    convert_norms_to_abs,
    convert_x_y_rel_cell_to_rel_image,
)
from model.yolov1 import calculate_class_confidence, apply_yolo_activation


def extract_and_convert_pred_bboxes(
    cfg: YOLOConfig, pred: torch.Tensor, img_offset: int = 0
):
    """
    Extracts bounding boxes from a batch of predictions and converts them from mid-points with normalized values to corner-points with absolute pixel values.

    Args:
        cfg (YOLOConfig): The configuration object.
        pred (torch.Tensor): A (N, S, S, 30) predictions tensor.
        img_offset (int): An offset to add to the image_idx to ensure unique indices across batches. This is primarily used for mAP.
            - When we need to extract bounding boxes across many batches primarily for mAP, we need to distinguish an image's index from one batch to the other batches, e.g., we can't let (batch_1 image_1) and (batch_2 image_1) have the same image_idx, this is where we use img_offset.

    Returns:
        torch.Tensor: A tensor of shape (N, 7) with format [image_idx, best_cls_idx, pc, x1, y1, x2, y2], where N is the number of predicted bounding boxes.
            - In corner-points with absolute pixel values.
    """
    S, B, C, IMAGE_SIZE = cfg.S, cfg.B, cfg.C, cfg.IMAGE_SIZE
    N = pred.shape[0]  # batch_size

    # --- 1. Separate box components and apply sigmoid to get probabilities and valid coordinates ---
    # 1.1: The raw prediction tensor needs activations applied.
    pred_activated = apply_yolo_activation(pred, C=C, B=B)

    # 1.2: Separate tensor data
    class_probs = pred_activated[..., :C]
    best_prob, best_class_idx = torch.max(class_probs, dim=-1)  # Shape: (N, S, S)

    #   Get the part of the tensor with box data -> [pc_1, x, y, w, h, pc_2, x, y, w, h]
    box_pred = pred_activated[..., C:]
    #   Reshape the box data to separate the B boxes. (N, S, S, B*5) -> (N, S, S, B, 5)
    box_pred = box_pred.reshape(N, S, S, B, 5)

    #   Extract components from the reshaped box data
    pc = box_pred[..., 0]  # Shape: (N, S, S, B) probability score
    box_coords = box_pred[..., 1:5]  # Shape: (N, S, S, B, 4) [x, y, w, h]

    #   Calculate final confidence scores.
    confidence = calculate_class_confidence(best_prob, pc)

    # --- 2: Convert from midpoints with normalized values to corner-points with absolute values ---
    # Create a grid of cell indices -> j_indices=row, i_indices=col
    j_indices = (
        torch.arange(S, device=pred_activated.device).repeat(N, S, 1).unsqueeze(-1)
    )
    i_indices = (
        torch.arange(S, device=pred_activated.device)
        .repeat(N, S, 1)
        .transpose(1, 2)
        .unsqueeze(-1)
    )

    #    Extract coordinates and ensure width/height are positive.
    x_rel_cell, y_rel_cell, w, h = box_coords.unbind(dim=-1)  # x_rel_cell: (2, S, S, B)
    w, h = torch.abs(w), torch.abs(h)  # (2, S, S, B)

    #   Convert (x, y) box midpoints from being relative to a cell, to be relative to the entire image.
    #   _mid_ stands for midpoints coordinates, _abs stands for absolute values.
    x_mid_abs = (x_rel_cell + j_indices) / S
    y_mid_abs = (y_rel_cell + i_indices) / S

    #   Convert to corner points and scale to absolute pixel values.
    x1 = (x_mid_abs - w / 2) * IMAGE_SIZE
    y1 = (y_mid_abs - h / 2) * IMAGE_SIZE
    x2 = (x_mid_abs + w / 2) * IMAGE_SIZE
    y2 = (y_mid_abs + h / 2) * IMAGE_SIZE

    # --- 3: Concatenate Tensors ---
    image_indices = (
        torch.arange(N, device=pred_activated.device)
        .view(N, 1, 1, 1)
        .expand_as(confidence)
        + img_offset
    )  # (2, S, S, B)

    # Expand best_class_idx to match the other tensors
    expanded_class_idx = best_class_idx.unsqueeze(-1).expand_as(
        confidence
    )  # (2, S, S, B)

    stacked_boxes = torch.stack(
        [image_indices.float(), expanded_class_idx.float(), confidence, x1, y1, x2, y2],
        dim=-1,
    )

    all_pred_boxes = stacked_boxes.reshape(-1, 7)
    return all_pred_boxes


def extract_and_convert_label_bboxes(
    cfg: YOLOConfig, labels: torch.Tensor, img_offset: int = 0
):
    """
    Extracts bounding boxes from a batch of labels and converts them from mid-points with normalized values to corner-points with absolute pixel values.

    Args:
        cfg (YOLOConfig): The configuration object.
        boxes_t (torch.Tensor): The (N, S, S, 30) label tensor from the dataset. Where N is the batch size.
        img_offset (int): An offset to add to the image_idx to ensure unique indices across batches. This is primarily used for mAP.
            - When we need to extract bounding boxes across many batches primarily for mAP, we need to distinguish an image's index from one batch to the other batches, e.g., we can't let (batch_1 image_1) and (batch_2 image_1) have the same image_idx, this is where we use img_offset.
    Returns:
        torch.Tensor: A tensor of shape (N, 7) with format [image_idx, class_idx, score, x1, y1, x2, y2], where N is the number of bounding boxes.
            - In corner-points with absolute pixel values.
    """
    S, C, IMAGE_SIZE = cfg.S, cfg.C, cfg.IMAGE_SIZE
    N = labels.shape[0]  # Batch size not hardcoded.

    # --- 1: Create Mask and indices ---
    #   Find where objects exists (pc=1)
    exists_mask = labels[..., C] == 1  # (N, S, S)

    #   Get the (image_idx, i, j) indices for all existing objects.
    indices = torch.nonzero(
        exists_mask, as_tuple=False
    )  # (num_boxes, 3) the num_boxes of bboxes that is in all the samples.

    if indices.numel() == 0:
        return torch.empty((0, 7), device=labels.device)

    #   Use the indices to gather the data for existing boxes.
    image_idxs = indices[:, 0] + img_offset  # (num_boxes)
    i_idxs = indices[:, 1]
    j_idxs = indices[:, 2]

    #   Gather data from the labels tensor using the mask
    box_data = labels[exists_mask]  # (num_boxes, CELL_NODES)

    # --- 2: Extract Components ---
    class_idx = box_data[:, :C].argmax(dim=-1)  # (num_boxes) e.g. ([0, 7, 7, 0, 0]) -> 0='person' 7='aeroplane' # fmt: skip

    #   For labels, confidence (pc) is always 1
    confidence = box_data[:, C]  # (num_boxes)
    coords = box_data[:, C + 1 : C + 5]  # (num_boxes, 4) [x, y, w, h]
    #   Separate the coords
    x_rel_cell, y_rel_cell, w, h = coords.unbind(dim=-1)  # x_rel_cell shape (num_boxes) contains all the x values from all the existing bounding boxes. # fmt: skip

    # --- 3: Convert to corner-points with absolute pixel values ---
    # 3.1. Convert the (x, y) from relative to cell to relative to image
    x_mid_norm, y_mid_norm = convert_x_y_rel_cell_to_rel_image(
        x_rel_cell, y_rel_cell, i_idxs, j_idxs, S
    )
    # 2.3. Convert from mid-point to corner-points (still normalized).
    x1_norm, y1_norm, x2_norm, y2_norm = convert_mid_to_corner(
        x_mid_norm, y_mid_norm, w, h
    )

    # 3.3. Convert from normalized corner-points to absolute pixel values.
    x1_abs = convert_norms_to_abs(x1_norm, IMAGE_SIZE)  # (num_boxes)
    y1_abs = convert_norms_to_abs(y1_norm, IMAGE_SIZE)
    x2_abs = convert_norms_to_abs(x2_norm, IMAGE_SIZE)
    y2_abs = convert_norms_to_abs(y2_norm, IMAGE_SIZE)

    all_label_boxes = torch.stack(
        [
            image_idxs.float(),
            class_idx.float(),
            confidence,
            x1_abs,
            y1_abs,
            x2_abs,
            y2_abs,
        ],
        dim=-1,
    )

    return all_label_boxes


# Test as module
#    python -m data.utils.bbox_extraction
def test():
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

    # --- Create a mini-batch with two samples for testing ---
    img1, label1 = d[0]
    # img2, label2 = d[1]
    # Stack labels to create a batch of size 2
    # label_batch = torch.stack([label1, label2])
    label_batch = torch.stack([label1])

    # --- Test batch-processed functions ---
    # print("--- Ground Truth Boxes (Batch) ---")
    # ground_truths = extract_and_convert_label_bboxes(cfg, label_batch)
    # for box in ground_truths:
    #     # Each box is [img_idx, class, score, x1, y1, x2, y2]
    #     print(
    #         f"Image {int(box[0])}: (Class, x1, y1, x2, y2) ->  [{box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}, {box[6]:.2f}]"
    #     )

    # --- Simulate a prediction batch
    # use the label as the prediction
    predictions = extract_and_convert_pred_bboxes(cfg, label_batch)
    print(f"Total boxes predicted in batch: {len(predictions)}")
    print("First 5 predicted boxes:")
    for box in predictions:
        # Each box is [img_idx, class, score, x1, y1, x2, y2]
        print(
            f"Image {int(box[0])}: (Class, x1, y1, x2, y2) ->  [{box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}, {box[6]:.2f}]"
        )


if __name__ == "__main__":
    test()
