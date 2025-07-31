"""Non-max-suppression"""

import torch
from torchvision.ops import nms as torch_nms

# My modules
from configs.config_loader import YOLOConfig


def nms(cfg: YOLOConfig, boxes: torch.Tensor):
    """
    OLD NMS function.

    Computes Non-Maximum-Suppression. NMS is a vital post-processing step in many computer vision tasks, particularly in object detection. It is used to clean-up predictions of object detection models by eliminating redundant bounding boxes and ensuring that each object is detected only once in a cell, i.e predicted bbox1 vs bbox2.

    NMS is typically computed on a per-image basis vs across an entire batch of images simultaneously.

    Args:
        cfg: Project configurations.
        boxes (torch.Tensor): Bounding boxes extracted from the models prediction.
            - A tensor of shape (N, 7) with format  [image_idx, best_cls_idx, pc, x1, y1, x2, y2]
            - In corner-points with absolute pixel values.
    Returns:
        torch.Tensor: Bounding boxes that have passed NMS, with the same format.
            - Returns an empty tensor if no boxes survive.
    """

    S, C, NMS_IOU_THRESHOLD, NMS_MIN_THRESHOLD = (
        cfg.S,
        cfg.C,
        cfg.NMS_IOU_THRESHOLD,
        cfg.NMS_MIN_THRESHOLD,
    )

    # BELOW IS ONLY FOR TESTING OVERFIT MODEL WITH LOW IOU AND MIN
    # NMS_IOU_THRESHOLD, NMS_MIN_THRESHOLD = 0.4, 0.2


    # ==> 1: Filter out low-confidence bounding boxes; tensors where probability scores is less than min_threshold.
    keep_mask = boxes[:, 2] > NMS_MIN_THRESHOLD
    bboxes = boxes[keep_mask]

    #   Return if no boxes survives.
    if bboxes.numel() == 0:
        print(
            "No bounding boxes has a probability score (pc) > NMS_MIN_THRESHOLD. The model is not good enough or something is wrong in code."
        )
        return torch.tensor([])

    # ==> 2: Perform NMS per class.
    final_boxes = []
    # Loop through the bounding boxes by class
    for cls in bboxes[:, 0].unique():
        # Get all boxes for the current class.
        cls_mask = bboxes[:, 0] == cls
        cls_bboxes = bboxes[cls_mask]

        # torch_nms need boxes in (x1, y1, x2, y2) format and their scores.
        box_coords = cls_bboxes[:, 3:7]
        box_scores = cls_bboxes[:, 1]

        # Used torch libraries NMS
        keep_indices = torch_nms(box_coords, box_scores, NMS_IOU_THRESHOLD)

        # Append the boxes that were kept
        final_boxes.append(cls_bboxes[keep_indices])

    if not final_boxes:
        print(
            "No bounding boxes survived NMS. The model is not good enough or something is wrong in code."
        )
        return torch.Tensor([])

    return torch.cat(final_boxes, dim=0)
