"""
Utils to convert coordinates from mid-point to corner-points and vice versa.
"""

import torch

# My modules
from configs.config_loader import load_config, YOLOConfig
from data.voc_dataset import VOCDataset
from data.utils.setup_transforms import setup_transforms


def extract_and_convert_pred_bboxes(cfg: YOLOConfig, pred: torch.Tensor):
    """
    Extracts bounding boxes from a batch of predictions and converts them from mid-points with normalized values to corner-points with absolute pixel values.

    Args:
        cfg (YOLOConfig): The configuration object.
        pred (torch.Tensor): A (N, S, S, 30) predictions tensor.

    Returns:
        torch.Tensor: A tensor of shape (N, 7) with format [image_idx, best_cls_idx, pc, x1, y1, x2, y2], where N is the number of bounding boxes.
            - In corner-points with absolute pixel values.
    """

    S, B, C, IMAGE_SIZE = cfg.S, cfg.B, cfg.C, cfg.IMAGE_SIZE
    N = pred.shape[0]  # batch_size

    # --- 1. Separate tensor data. ---
    #    Separate class probabilities from box predictions.
    class_probs = pred[..., :C]  # Shape: (N, S, S, C)
    best_prob, best_class_idx = torch.max(class_probs, dim=-1)  # Shape: (N, S, S)

    #   Get the part of the tensor with box data -> [pc_1, x, y, w, h, pc_2, x, y, w, h]
    box_pred = pred[..., C:]  # Shape: (N, S, S, B*5)

    #   Reshape the box data to separate the B boxes. (N, S, S, B*5) -> (N, S, S, B, 5)
    box_pred = box_pred.reshape(N, S, S, B, 5)

    #   Extract components from the reshaped box data
    pc = box_pred[..., 0]  # Shape: (N, S, S, B) probability score
    box_coords = box_pred[..., 1:5]  # Shape: (N, S, S, B, 4) [x, y, w, h]

    #   Calculate final confidence scores
    # Expand best_prob to match the shape of objectness for broadcasting
    confidence = pc * best_prob.unsqueeze(-1)  # Shape: (N, S, S, B)
    """Note: this line:
                confidence = pc * best_prob.unsqueeze(-1)
                    ↓
                    Is described in the paper at 2. Unified Detection section:
                        "At test time we multiply the conditional class probabilities and the individual box confidence
                        predictions (Formula can't be displayed in python look into paper) which gives us class-specific
                        confidence scores for each box. These scores encode both the  probability of that class appearing
                        in the box and how well the predicted box fits the object."
    """

    # --- 2: Convert from midpoints with normalized values to corner-points with absolute values ---
    # Create a grid of cell indices -> j_indices=row, i_indices=col
    j_indices = torch.arange(S, device=pred.device).repeat(N, S, 1).unsqueeze(-1)
    i_indices = (
        torch.arange(S, device=pred.device)
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
        torch.arange(N, device=pred.device).view(N, 1, 1, 1).expand_as(confidence)
    )  # (2, S, S, B)

    # Expand best_class_idx to match the other tensors
    expanded_class_idx = best_class_idx.unsqueeze(-1).expand_as(
        confidence
    )  # (2, S, S, B)

    stacked_boxes = torch.stack(
        [image_indices.float(), expanded_class_idx.float(), confidence, x1, y1, x2, y2],
        dim=-1,
    )

    all_pred_boxes = stacked_boxes.reshape(-1, 7)  # .tolist()
    return all_pred_boxes


def extract_and_convert_label_bboxes(cfg: YOLOConfig, labels: torch.Tensor):
    """
    Extracts bounding boxes from a batch of labels and converts them from mid-points with normalized values to corner-points with absolute pixel values.

    Args:
        cfg (YOLOConfig): The configuration object.
        boxes_t (torch.Tensor): The (N, S, S, 30) label tensor from the dataset. Where N is the batch size.

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
        return []

    #   Use the indices to gather the data for existing boxes
    image_idxs = indices[:, 0]  # (num_boxes)
    i_idxs = indices[:, 1]
    j_idxs = indices[:, 2]

    #   Gather data  from the labels tensor using the mask
    box_data = labels[exists_mask]  # (num_boxes, CELL_NODES)

    # --- 2: Extract Components ---
    class_idx = box_data[:, :C].argmax(
        dim=-1
    )  # (num_boxes) e.g. ([0, 7, 7, 0, 0]) -> 0='person' 7='aeroplane'
    #   For labels, confidence (pc) is always 1
    confidence = box_data[:, C]  # (num_boxes)
    coords = box_data[:, C + 1 : C + 5]  # (num_boxes, 4) [x, y, w, h]
    #   Separate the coords
    x_rel_cell, y_rel_cell, w, h = coords.unbind(
        dim=-1
    )  # x_rel_cell shape (num_boxes) contains all the x values from all the existing bounding boxes

    # --- 3: Convert to corners points with absolute pixel values ---
    #   Convert (x, y) box midpoints from being relative to a cell, to be relative to the entire image.
    x_mid_abs = (x_rel_cell + j_idxs) / S  # (num_boxes)
    y_mid_abs = (y_rel_cell + i_idxs) / S

    #   Convert to corner points and scale to absolute pixel values
    x1 = (x_mid_abs - w / 2) * IMAGE_SIZE  # (num_boxes)
    y1 = (y_mid_abs - h / 2) * IMAGE_SIZE
    x2 = (x_mid_abs + w / 2) * IMAGE_SIZE
    y2 = (y_mid_abs + h / 2) * IMAGE_SIZE

    # --- 4: Concatenate all ---
    all_label_boxes = torch.stack(
        [image_idxs.float(), class_idx.float(), confidence, x1, y1, x2, y2], dim=-1
    )  # .tolist()

    return all_label_boxes


# Test as module
#    python -m data.utils.bbox_utils
def test():
    cfg = load_config("config_voc_dataset.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)
    d = VOCDataset(
        cfg=cfg,
        which_dataset=cfg.TRAIN_DIR_NAME,
        num_samples=cfg.NUM_TRAIN_SAMPLES,
        transforms=t,
    )

    # --- Create a mini-batch with two samples for testing ---
    img1, label1 = d[0]
    img2, label2 = d[1]
    # Stack labels to create a batch of size 2
    label_batch = torch.stack([label1, label2])

    # --- Test batch-processed functions ---
    # print("--- Ground Truth Boxes (Batch) ---")
    ground_truths = extract_and_convert_label_bboxes(cfg, label_batch)
    # for box in ground_truths:
    #     # Each box is [img_idx, class, score, x1, y1, x2, y2]
    #     print(f"Image {int(box[0])}: Class {int(box[1])}, Coords [{box[2]:.2f}, {box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}]")

    # --- Simulate a prediction batch (can be random for testing shape)
    # pred_batch = torch.rand(2, cfg.S, cfg.S, cfg.C + cfg.B * 5)
    # print("\n--- Predicted Boxes (Batch) ---")
    # predictions = extract_and_convert_pred_bboxes(cfg, pred_batch)
    # print(f"Total boxes predicted in batch: {len(predictions)}")
    # print("First 5 predicted boxes:")
    # for box in predictions[:5]:
    #     print(
    #         f"Image {int(box[0])}: Class {int(box[1])}, Conf {box[2]:.2f}, Coords [{box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}, {box[6]:.2f}]"
    #     )


if __name__ == "__main__":
    test()


"""Below two functions are un-vectorized and only work for one sample/tensor"""

# def extract_and_convert_pred_bboxes(cfg: YOLOConfig, pred: torch.Tensor):
#     """
#     Extracts bounding boxes from a prediction tensor and converts them from mid-points with normalized values to corner-points with absolute pixel values.

#     Args:
#         cfg (YOLOConfig): The configuration object.
#         pred (torch.Tensor): A ( S, S, 30) predictions tensor.

#     Returns:
#         torch.Tensor: A tensor of shape (N, 6) with format [best_cls_idx, pc, x1, y1, x2, y2]
#             - In corner-points with absolute pixel values.
#     """

#     S, B, C, IMAGE_SIZE = cfg.S, cfg.B, cfg.C, cfg.IMAGE_SIZE

#     all_boxes = []

#     # Iterate over each cell in the SxS grid:
#     for i in range(S):
#         for j in range(S):
#             # Iterate over bounding boxes per cell.
#             for b in range(B):
#                 # Get the objectness score (pc) for the bth bounding box.
#                 pc_idx = C + b * 5
#                 pc = pred[i, j, pc_idx]

#                 # --- Box Extraction And Conversion ---
#                 #   Get the class probabilities and find the most likely class
#                 class_probs = pred[i, j, :C]  # index (0-19)
#                 best_prob, best_class_idx = torch.max(class_probs, dim=-1)

#                 #   Calculate the final confidence score for this detection
#                 confidence = pc * best_prob
#                 """Note: this line:
#                                 confidence = pc * best_prob
#                         is described in the paper as:
#                                 'At test time we multiply the conditional class probabilities and the individual box confidence
#                                  predictions, -> [Formula can't be displayed in python look into paper]
#                                 which gives us class-specific confidence scores for each box. These scores encode both the
#                                 probability of that class appearing in the box and how well the predicted box fits the
#                                 object.'
#                 """

#                 #   Extract the coordinates (x, y, w, h) for the b-th box.
#                 coords_idx = pc_idx + 1
#                 box_coords = pred[i, j, coords_idx : coords_idx + 4]
#                 # The x, y are still relative to the cell
#                 x_rel_cell, y_rel_cell, w, h = box_coords
#                 # make sure w and h aren't negatives.
#                 w, h = torch.abs(w), torch.abs(h)

#                 # --- Convert (x, y) box midpoints to be relative to the image.
#                 # _mid_ stands for midpoints coordinates.
#                 x_mid_abs = (x_rel_cell + j) / S
#                 y_mid_abs = (y_rel_cell + i) / S

#                 # --- Convert midpoint format to corner point format and scale from normalized percentage to Absolute pixel values.
#                 x1 = (x_mid_abs - w / 2) * IMAGE_SIZE
#                 y1 = (y_mid_abs - h / 2) * IMAGE_SIZE
#                 x2 = (x_mid_abs + w / 2) * IMAGE_SIZE
#                 y2 = (y_mid_abs + h / 2) * IMAGE_SIZE

#                 all_boxes.append(
#                     [
#                         best_class_idx.item(),
#                         confidence.item(),
#                         x1.item(),
#                         y1.item(),
#                         x2.item(),
#                         y2.item(),
#                     ]
#                 )
#     return torch.tensor(all_boxes)


# def extract_and_convert_label_bboxes(boxes_t: torch.Tensor, cfg: YOLOConfig):
#     """
#     Extracts bounding boxes from a label tensor and converts them from mid-points with normalized values to corner-points with absolute pixel values.

#     Args:
#         boxes_t (torch.Tensor): The (S, S, 30) label tensor from the dataset.
#         cfg (YOLOConfig): The configuration object.

#     Returns:
#         torch.Tensor: A tensor of shape (N, 6) with format [cls_idx, score, x1, y1, x2, y2]
#             - In corner-points with absolute pixel values.
#     """
#     S, C = cfg.S, cfg.C
#     IMAGE_SIZE = cfg.IMAGE_SIZE

#     # Create a boolean mask for where an object exists (pc_1 == 1)
#     exists_mask = boxes_t[:, :, C] == 1

#     # Get the (row, col) or (i, j) indices of cells containing objects
#     indices = torch.nonzero(exists_mask, as_tuple=False)

#     all_abs_boxes = []

#     if indices.numel() == 0:
#         print("\nNo bounding boxes found!\n")
#         return torch.empty((0, 6))

#     for idx_pair in indices:
#         # Use the i and j cell indice to correctly convert.
#         i, j = idx_pair[0].item(), idx_pair[1].item()
#         box_data = boxes_t[i, j]

#         # Extract data for the first bounding box predictor
#         class_idx = box_data[:C].argmax().item()
#         pc, x_rel, y_rel, width, height = box_data[C : C + 5].tolist()

#         # Step 2: Reconstruct absolute midpoint (relative to image 0-1)
#         x_mid_abs = (x_rel + j) / S
#         y_mid_abs = (y_rel + i) / S

#         # Width and height are already relative to the image
#         w_abs = width
#         h_abs = height

#         # Step 3: Convert to corner-point format (relative to image 0-1)
#         x1_rel = x_mid_abs - w_abs / 2
#         y1_rel = y_mid_abs - h_abs / 2
#         x2_rel = x_mid_abs + w_abs / 2
#         y2_rel = y_mid_abs + h_abs / 2

#         # Step 4: Scale to absolute pixel values
#         x1 = x1_rel * IMAGE_SIZE
#         y1 = y1_rel * IMAGE_SIZE
#         x2 = x2_rel * IMAGE_SIZE
#         y2 = y2_rel * IMAGE_SIZE

#         all_abs_boxes.append([class_idx, pc, x1, y1, x2, y2])

#     return torch.tensor(all_abs_boxes)


# # Test as module
# #    python -m data.utils.bbox_utils
# def test():
#     cfg = load_config("config_voc_dataset.yaml")
#     t = setup_transforms(cfg.IMAGE_SIZE)
#     d = VOCDataset(cfg=cfg, which_dataset=cfg.TRAIN_DIR_NAME, transforms=t)
#     img, label = d.__getitem__(2)
#     # --- Test extract_and_convert_bboxes() -> with Tensors are in mid-point format and normalized percentage.
#     print(extract_and_convert_label_bboxes(label, cfg))


# if __name__ == "__main__":
#     test()
