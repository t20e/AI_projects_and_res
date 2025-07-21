"""Non-Max Suppression"""

import torch
from torchvision.ops import nms


def non_max_suppression(pred_bboxes, config):
    """
    Computes NMS by filtering out overlapping bboxes per class.

    Note:
        Non-Maximum Suppression (NMS) is a vital post-processing step in many computer vision tasks, particularly in object detection. It is used to refine predictions of object detection models by eliminating redundant bounding boxes and ensuring that each object is detected only once.

    Args:
        pred_bboxes (Tensor): (N, 9) [ i, j, b, class_idx, pc, x, y, w, h]
            - x,y are cell-relative ∈ [0,1];  w,h are image-relative ∈ [0,1].

        config : Namespace configurations.
            Project configurations.

    Variables:
        IOU_threshold (float) : the iou threshold when comparing bounding boxes for NMS.

        min_threshold (float) : the threshold to remove bounding boxes with a low confidence score.
    Returns:
        Tensor : (M, 9) with filtered bboxes.
    """

    S, IOU_THRESHOLD, MIN_THRESHOLD = (
        config.S,
        config.IOU_THRESHOLD,
        config.MIN_THRESHOLD,
    )

    # === 1: Filter out low-confidence bboxes; tensors where pc/probability_score (column 4) is less than min_threshold.
    keep_mask = pred_bboxes[:, 4] > MIN_THRESHOLD
    bboxes = pred_bboxes[keep_mask]

    #   Return if no boxes survives.
    if bboxes.numel() == 0:
        print(
            "No bounding boxes has a probability score (pc) > MIN_THRESHOLD. The model is not good enough or something is wrong in code."
        )
        return bboxes

    # === 2: Convert midpoint coords (x, y, w, h) -> corner-point coordinates (x1, y1, x2, y2).
    i, j = bboxes[:, 0], bboxes[:, 1]
    cx = (j + bboxes[:, 5]) / S
    cy = (i + bboxes[:, 6]) / S
    w, h = bboxes[:, 7], bboxes[:, 8]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    corners = torch.stack([x1, y1, x2, y2], dim=1)

    # === 3: Perform class-wise NMS
    final_idx = []
    for cls in bboxes[:, 3].unique():
        cls_mask = bboxes[:, 3] == cls
        cls_boxes = corners[cls_mask]
        cls_scores = bboxes[cls_mask, 4]  # pc
        # TODO I need to do the NMS line below myself instead of calling from a library
        keep = nms(cls_boxes, cls_scores, IOU_THRESHOLD)
        base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
        final_idx.append(base_idx[keep])

    final_idx = torch.cat(final_idx)
    return bboxes[final_idx]


# --
# --
# TODO del if not needed
#  NOTE: =============================================
#   - Special case function below only meant to be called by the non_max_suppression() function.
#   - while below seems intuitive its better to keep bboxes shape as the grid structure instead of [(N, 9) that the below is expecting its args tensor shapes to be] when using for Loss function. Above function handles that.
#   - Unfortunately had to use two IOU functions for two different use cases.
# def iou_for_non_max_suppression(chosen_bbox: torch.Tensor, rest_bbox: torch.Tensor):
#     """
#     THIS FUNCTION IS SPECIAL CASE ONLY MEANT TO BE USED BY THE NON_MAX_SUPPRESSION() function. Computes Intersection Over Union between one bbox and batch of bbox.

#     Note:
#         This function requires that the bbox coordinates format are in mid-point format. And that they all be passed in with the same class_idx.

#     Args:
#         bbox (tensor): shape: (9). [i, j, b, class_idx, pc, x, y, w, h].
#         coords (tensor): Shape (N, 9). [i, j, b, class_idx, pc, x, y, w, h].

#     Returns:
#         tensor : IOU values.
#     """
#     # --- 1: Extract and convert coordinates to corner-points format
#     x1 = rest_bbox[:, 5] - rest_bbox[:, 7] / 2
#     y1 = rest_bbox[:, 6] - rest_bbox[:, 8] / 2
#     x2 = rest_bbox[:, 5] + rest_bbox[:, 7] / 2
#     y2 = rest_bbox[:, 6] + rest_bbox[:, 8] / 2

#     # Do it for chosen_bbox
#     cx, cy, cw, ch = chosen_bbox[5:9]
#     cx1, cy1 = cx - cw / 2, cy - ch / 2
#     cx2, cy2 = cx + cw / 2, cy + ch / 2
#     chosen_bbox_area = (cx2 - cx1) * (cy2 - cy1)

#     # --- 2: Calculate IOU
#     box_area = (x2 - x1) * (y2 - y1)
#     inter_x1 = torch.max(cx1, x1)
#     inter_y1 = torch.max(cy1, y1)
#     inter_x2 = torch.min(cx2, x2)
#     inter_y2 = torch.min(cy2, y2)

#     inter_w = (inter_x2 - inter_x1).clamp(min=0)
#     inter_h = (inter_y2 - inter_y1).clamp(min=0)
#     inter_area = inter_w * inter_h

#     union_area = chosen_bbox_area + box_area - inter_area
#     iou = inter_area / (union_area + 1e-6)  # example print tensor([0.8223, 0.0000])
#     return iou
