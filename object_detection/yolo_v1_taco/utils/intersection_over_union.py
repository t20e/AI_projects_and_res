"""IOU"""
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


 
def intersection_over_union(chosen_bbox:torch.Tensor, rest_bbox:torch.Tensor):
    """
    Computes Intersection Over Union between one bbox and batch of bbox. vectorized.
    
    Note: 
        This function requires that the bbox coordinates format are in mid-point format. And that they all be passed in with the same class_idx.
    
    Args:
        bbox (tensor): shape: (9). [i, j, b, class_idx, pc, x, y, w, h].
        coords (tensor): Shape (N, 9). [i, j, b, class_idx, pc, x, y, w, h].
    
    Returns:
        tensor : IOU values.
    """
    # --- 1: Extract and convert coordinates to corner-points format
    x1 = rest_bbox[:, 5] - rest_bbox[:, 7] / 2
    y1 = rest_bbox[:, 6] - rest_bbox[:, 8] / 2
    x2 = rest_bbox[:, 5] + rest_bbox[:, 7] / 2
    y2 = rest_bbox[:, 6] + rest_bbox[:, 8] / 2

    # Do it for chosen_bbox
    cx, cy, cw, ch = chosen_bbox[5:9]
    cx1, cy1 = cx - cw / 2, cy - ch / 2
    cx2, cy2 = cx + cw / 2, cy + ch / 2
    chosen_bbox_area = (cx2 - cx1) * (cy2 - cy1)

    # --- 2: Calculate IOU
    box_area = (x2 - x1) * (y2 - y1)
    inter_x1 = torch.max(cx1, x1)
    inter_y1 = torch.max(cy1, y1)
    inter_x2 = torch.min(cx2, x2)
    inter_y2 = torch.min(cy2, y2)
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = chosen_bbox_area + box_area - inter_area
    iou = inter_area / (union_area + 1e-6)# example print tensor([0.8223, 0.0000])
    return iou