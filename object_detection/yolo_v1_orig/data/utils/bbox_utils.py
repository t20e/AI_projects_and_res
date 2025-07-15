"""
Utils to convert coordinates from mid-point to corner-points and vice versa.
"""

import torch

# My modules
from configs.config_loader import load_config, YOLOConfig
from data.voc_dataset import VOCDataset
from data.utils.setup_transforms import setup_transforms


# NOTE: TODO I think this function can only be used for ploting, maybe renamed to plotting
def extract_and_convert_bboxes(boxes_t:torch.Tensor, cfg:YOLOConfig):
    """
    Extracts bounding boxes from a tensor and converts them from mid-points with normalized values to corner-points with absolute pixel values.

    Args:
        boxes_t (torch.Tensor): The (S, S, 30) label tensor from the dataset.
        cfg (YOLOConfig): The configuration object.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) with format [cls_idx, score, x1, y1, x2, y2]
                      in absolute pixel values.
    """
    S, C = cfg.S, cfg.C
    IMAGE_SIZE = cfg.IMAGE_SIZE

    # Create a boolean mask for where an object exists (pc_1 == 1)
    exists_mask = boxes_t[:, :, C] == 1

    # Get the (row, col) or (i, j) indices of cells containing objects
    indices = torch.nonzero(exists_mask, as_tuple=False)

    all_abs_boxes = []

    if indices.numel() == 0:
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
    print(extract_and_convert_bboxes(label, cfg))


# test()
