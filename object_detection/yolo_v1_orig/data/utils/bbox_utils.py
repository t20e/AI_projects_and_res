"""
Utils to convert coordinates from mid-point to corner-points and vice versa.
"""

import torch

# My modules
from configs.config_loader import load_config, YOLOConfig
from data.voc_dataset import VOCDataset
from data.utils.setup_transforms import setup_transforms

# Note the two extract functions below could be made more dry and vectorized but for simplicity I did it like so.


def extract_and_convert_pred_bboxes(cfg: YOLOConfig, pred: torch.Tensor):
    """
    Extracts bounding boxes from a prediction tensor and converts them from mid-points with normalized values to corner-points with absolute pixel values.

    Args:
        cfg (YOLOConfig): The configuration object.
        pred (torch.Tensor): The (batch, S, S, 30) predictions tensor.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) with format [best_cls_idx, pc, x1, y1, x2, y2]
            - In corner-points with absolute pixel values.
    """

    S, B, C, IMAGE_SIZE = cfg.S, cfg.B, cfg.C, cfg.IMAGE_SIZE

    all_boxes = []

    # Iterate over each cell in the SxS grid:
    for i in range(S):
        for j in range(S):
            # Iterate over bounding boxes per cell.
            for b in range(B):
                # Get the objectness score (pc) for the bth bounding box.
                pc_idx = C + b * 5
                pc = pred[i, j, pc_idx]

                # --- Box Extraction And Conversion ---
                #   Get the class probabilities and find the most likely class
                class_probs = pred[i, j, :C]  # index (0-19)
                best_prob, best_class_idx = torch.max(class_probs, dim=-1)

                #   Calculate the final confidence score for this detection
                confidence = pc * best_prob  
                """Note: this line:
                                confidence = pc * best_prob 
                        is described in the paper as:
                                'At test time we multiply the conditional class probabilities and the individual box confidence
                                 predictions, -> [Formula can't be displayed in python look into paper]
                                which gives us class-specific confidence scores for each box. These scores encode both the 
                                probability of that class appearing in the box and how well the predicted box fits the
                                object.'
                """

                #   Extract the coordinates (x, y, w, h) for the b-th box.
                coords_idx = pc_idx + 1
                box_coords = pred[i, j, coords_idx : coords_idx + 4]
                # The x, y are still relative to the cell
                x_rel_cell, y_rel_cell, w, h = box_coords
                # make sure w and h aren't negatives.
                w, h = torch.abs(w), torch.abs(h)

                # --- Convert (x, y) box midpoints to be relative to the image.
                # _mid_ stands for midpoints coordinates.
                x_mid_abs = (x_rel_cell + j) / S
                y_mid_abs = (y_rel_cell + i) / S

                # --- Convert midpoint format to corner point format and scale from normalized percentage to Absolute pixel values.
                x1 = (x_mid_abs - w / 2) * IMAGE_SIZE
                y1 = (y_mid_abs - h / 2) * IMAGE_SIZE
                x2 = (x_mid_abs + w / 2) * IMAGE_SIZE
                y2 = (y_mid_abs + h / 2) * IMAGE_SIZE

                all_boxes.append(
                    [
                        best_class_idx.item(),
                        confidence.item(),
                        x1.item(),
                        y1.item(),
                        x2.item(),
                        y2.item(),
                    ]
                )
    return torch.tensor(all_boxes)


def extract_and_convert_label_bboxes(boxes_t: torch.Tensor, cfg: YOLOConfig):
    """
    Extracts bounding boxes from a label tensor and converts them from mid-points with normalized values to corner-points with absolute pixel values.

    Args:
        boxes_t (torch.Tensor): The (S, S, 30) label tensor from the dataset.
        cfg (YOLOConfig): The configuration object.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) with format [cls_idx, score, x1, y1, x2, y2]
            - In corner-points with absolute pixel values.
    """
    S, C = cfg.S, cfg.C
    IMAGE_SIZE = cfg.IMAGE_SIZE

    # Create a boolean mask for where an object exists (pc_1 == 1)
    exists_mask = boxes_t[:, :, C] == 1

    # Get the (row, col) or (i, j) indices of cells containing objects
    indices = torch.nonzero(exists_mask, as_tuple=False)

    all_abs_boxes = []

    if indices.numel() == 0:
        print("\nNo bounding boxes found!\n")
        return torch.empty((0, 6))

    for idx_pair in indices:
        # Use the i and j cell indice to correctly convert.
        i, j = idx_pair[0].item(), idx_pair[1].item()
        box_data = boxes_t[i, j]

        # Extract data for the first bounding box predictor
        class_idx = box_data[:C].argmax().item()
        pc, x_rel, y_rel, width, height = box_data[C : C + 5].tolist()

        # Step 2: Reconstruct absolute midpoint (relative to image 0-1)
        x_mid_abs = (x_rel + j) / S
        y_mid_abs = (y_rel + i) / S

        # Width and height are already relative to the image
        w_abs = width
        h_abs = height

        # Step 3: Convert to corner-point format (relative to image 0-1)
        x1_rel = x_mid_abs - w_abs / 2
        y1_rel = y_mid_abs - h_abs / 2
        x2_rel = x_mid_abs + w_abs / 2
        y2_rel = y_mid_abs + h_abs / 2

        # Step 4: Scale to absolute pixel values
        x1 = x1_rel * IMAGE_SIZE
        y1 = y1_rel * IMAGE_SIZE
        x2 = x2_rel * IMAGE_SIZE
        y2 = y2_rel * IMAGE_SIZE

        all_abs_boxes.append([class_idx, pc, x1, y1, x2, y2])

    return torch.tensor(all_abs_boxes)


# Test as module
# python -m data.utils.bbox_utils
def test():
    cfg = load_config("yolov1.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)
    d = VOCDataset(cfg, t)
    img, label = d.__getitem__(2)
    # --- Test extract_and_convert_bboxes() -> with Tensors are in mid-point format and normalized percentage.
    print(extract_and_convert_label_bboxes(label, cfg))


if __name__ == "__main__":
    test()
