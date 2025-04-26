
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def intersection_over_union(pred_bboxes, label_bboxes):
    """
    Calculates Intersection Over Union, i.e compares predicted bounding boxes to the true bounding boxes.
    
    Note: You can calculate IOU from "Mid-point" or "corner-point" format.
        Mid-point format: [x_center, y_center, height, width]
        Corner-point format: [x_min, y_min, x_max, y_max]
        This function below only uses the mid-point format.
    
    Parameters
    ----------
        pred_bboxes : (tensor)
            Model bounding boxes predictions (Batch_size, 4)
        label_bboxes : (tensor)
            True bounding boxes (Batch_size, 4)
    
    Returns
    -------
        tensor : IOU for all bboxes
    """
    # Mid-point formulas
    box1_x1 = pred_bboxes[..., 0:1] - pred_bboxes[..., 2:3] / 2
    box1_y1 = pred_bboxes[..., 1:2] - pred_bboxes[..., 3:4] / 2
    box1_x2 = pred_bboxes[..., 0:1] + pred_bboxes[..., 2:3] / 2
    box1_y2 = pred_bboxes[..., 1:2] + pred_bboxes[..., 3:4] / 2
    box2_x1 = label_bboxes[..., 0:1] - label_bboxes[..., 2:3] / 2
    box2_y1 = label_bboxes[..., 1:2] - label_bboxes[..., 3:4] / 2
    box2_x2 = label_bboxes[..., 0:1] + label_bboxes[..., 2:3] / 2
    box2_y2 = label_bboxes[..., 1:2] + label_bboxes[..., 3:4] / 2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)
    
    # .clamp(0) is for the case where they dont intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / ( box1_area + box2_area - intersection + 1e-6)