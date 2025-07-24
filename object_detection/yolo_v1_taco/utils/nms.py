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
        keep = nms(cls_boxes, cls_scores, IOU_THRESHOLD)
        base_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
        final_idx.append(base_idx[keep])

    final_idx = torch.cat(final_idx)
    return bboxes[final_idx]


# --
# -- 