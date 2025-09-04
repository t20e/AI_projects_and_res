import torch

def intersection_over_union(pred_bboxes, label_bboxes, box_format="midpoint"):
    """
    Calculates intersection over union

    Note:
        There are two types of bounding box (bbox) coordinate formats:  mid-point and corner-points.

            Mid-point format: (x, y, w, h)
                - x, y represent the center point of the bounding box.
                - In YOLOv1, x and y are relative to the grid cell that predicts the object.
                    That means x, y are in the range [0, 1], where (0,0) is the top-left of the cell and (1,1) is the bottom-right of the same cell.
                - w and h represent the width and height of the bounding box, relative to the entire image!

            Corner-points format: (x_min, y_min, x_max, y_max)
                - x_min, y_min represent the top-left corner of the bounding box.
                - x_max, y_max represent the bottom-right corner.
                - These coordinates are typically relative to the entire image and are often derived from mid-point format during postprocessing steps such as IoU calculation, visualization, or NMS.

        YOLOv1 uses mid-point format for its predictions, however its simpler to use corner-points for NMS, and IOU calculations. They are easily convertible.

    Args:
        pred_bboxes (tensor): Predictions of bounding boxes (batch_size, 4)
        label_bboxes (tensor): True bounding boxes (batch_size, 4)
        box_format (str) : "midpoint" or "corners" used to specify layout.
            - "midpoint": [x, y, w, h]
            - "corners": [x1, y1, x2, y2]
    
    Returns:
        tensor: Intersection over union for all examples in a batch.
    """

    # --- 1: Convert "midpoints" coordinates to "corners"-points, its easier to calculates IOU with corner points.
    if box_format == "midpoint":
        box1_x1 = pred_bboxes[..., 0:1] - pred_bboxes[..., 2:3] / 2 # pred_bboxes[..., 0:1] is the midpoint, pred_bboxes[..., 2:3] is the width of that bbox, so if we divide it by 2, then well get the top-left x1 coordinate
        box1_y1 = pred_bboxes[..., 1:2] - pred_bboxes[..., 3:4] / 2
        box1_x2 = pred_bboxes[..., 0:1] + pred_bboxes[..., 2:3] / 2
        box1_y2 = pred_bboxes[..., 1:2] + pred_bboxes[..., 3:4] / 2
        box2_x1 = label_bboxes[..., 0:1] - label_bboxes[..., 2:3] / 2
        box2_y1 = label_bboxes[..., 1:2] - label_bboxes[..., 3:4] / 2
        box2_x2 = label_bboxes[..., 0:1] + label_bboxes[..., 2:3] / 2
        box2_y2 = label_bboxes[..., 1:2] + label_bboxes[..., 3:4] / 2

    # --- 2: Grab the corner points
    elif box_format == "corners":
        box1_x1 = pred_bboxes[..., 0:1]  # the ... slices but keeps the same dimensions.
        box1_y1 = pred_bboxes[..., 1:2]
        box1_x2 = pred_bboxes[..., 2:3]
        box1_y2 = pred_bboxes[..., 3:4]
        box2_x1 = label_bboxes[..., 0:1]
        box2_y1 = label_bboxes[..., 1:2]
        box2_x2 = label_bboxes[..., 2:3]
        box2_y2 = label_bboxes[..., 3:4]
    
    # --- 3: Calculate the intersection. Get the coordinates of the overlapping box where pred_bboxes and label_bboxes overlap (the intersection).
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # --- 4: Calculate the intersection Area. The box where (box1, box2) intersect.
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # the .clamp(0) is only for cases when the bounding boxes don't intersect, if they dont it replaces all negative values with 0.

    # --- 5: Calculating the Union Area. The union is the total area covered by both (box1, box2)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1)) # the abs() is to make sure we dont have negatives.
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection
    
    # --- 6: Calculate Final IoU scores.
    epsilon = 1e-6 # Add a small epsilon to avoid division by zero
    return intersection / (union + epsilon)
