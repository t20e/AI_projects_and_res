"""Non-max-suppression"""

import torch

# My modules
from configs.config_loader import YOLOConfig
from utils.IoU import IoU_one_to_many_mapping


def nms(cfg: YOLOConfig, boxes: torch.Tensor):
    """
    Computes Non-Maximum-Suppression efficiently across a batch of images. NMS is a vital post-processing step in many computer vision tasks, particularly in object detection. It is used to clean-up predictions of object detection models by eliminating redundant bounding boxes and ensuring that each object is detected only once in a cell, i.e predicted bbox1 vs bbox2.

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

    ## NOTE: BELOW LINE IS ONLY FOR TESTING OVERFIT MODEL WITH LOW THRESHOLDS
    ## NMS_IOU_THRESHOLD, NMS_MIN_THRESHOLD = 0.6, 0.1

    # --- 1. Filter out low-confidence bounding boxes globally.
    keep_mask_conf = boxes[:, 2] > NMS_MIN_THRESHOLD
    filtered_boxes = boxes[keep_mask_conf]

    if filtered_boxes.numel() == 0:
        print(
            "No bounding boxes has a probability score (pc) > NMS_MIN_THRESHOLD. The model is not good enough or something is wrong in code."
        )
        return torch.empty((0, 7), device=boxes.device, dtype=boxes.dtype)

    final_nms_boxes = []

    # --- 2. Iterate through unique image IDs from entire dataset.
    # NMS fundamentally needs to be applied per image and per class.
    for img_idx in filtered_boxes[:, 0].unique():
        img_mask = filtered_boxes[:, 0] == img_idx
        img_boxes = filtered_boxes[img_mask]

        # Iterate through each unique class IDs [c1-c20] of bboxes in this image.
        for class_idx in img_boxes[:, 1].unique():
            # Get all boxes for the current class in the current image
            class_mask = img_boxes[:, 1] == class_idx
            class_img_boxes = img_boxes[class_mask]

            if len(class_img_boxes) <= 1:
                final_nms_boxes.append(class_img_boxes)
                continue

            # Sort bounding boxes by (pc) score in descending order (highest confidence first).
            scores, order = class_img_boxes[:, 2].sort(descending=True)
            sorted_boxes = class_img_boxes[order]

            #   Keep track of which boxes to retain with this class ID.
            keep_mask = torch.ones(
                len(sorted_boxes), dtype=torch.bool, device=boxes.device
            )

            # Loop thru each box that hasn't been (suppressed or okayed) yet.
            for i in range(len(sorted_boxes)):
                if not keep_mask[i]:  # If this box has already been suppressed, skip
                    continue
                # Get the current box (the one with highest remaining confidence score).
                current_box_coords = sorted_boxes[i, 3:7]

                # Compare it against all *subsequent* boxes in the sorted list that are not yet suppressed.
                subsequent_active_mask = keep_mask.clone()
                subsequent_active_mask[: i + 1] = False

                if not subsequent_active_mask.any():
                    break  # No more candidates to compare against

                # Get the coordinates of these boxes.
                subsequent_active_coords = sorted_boxes[subsequent_active_mask, 3:7]
                # print("\n\nmain", current_box_coords)
                # print("other", subsequent_active_coords)

                # Calculate IoU between the current and all subsequent boxes.
                ious = IoU_one_to_many_mapping(
                    current_box_coords,
                    subsequent_active_coords,
                    box_format="corners",
                )
                # print(ious)

                # Find which of these boxes overlap too much.
                suppress_mask_relative = ious > cfg.NMS_IOU_THRESHOLD
                # print("suppress_mask_relative", suppress_mask_relative)

                # Get the global indices of all boxes that are subsequent AND active
                global_indices_of_subsequent_active = torch.where(
                    subsequent_active_mask
                )[0]

                # Use the suppress_mask_relative to filter these global indices
                indices_to_suppress = global_indices_of_subsequent_active[
                    suppress_mask_relative
                ]
                # print("indices to suppress:", indices_to_suppress)

                if indices_to_suppress.numel() > 0:
                    keep_mask[indices_to_suppress] = False

            # Append the boxes that survived NMS for this image and class
            final_nms_boxes.append(sorted_boxes[keep_mask])

    if not final_nms_boxes:
        print(
            "No bounding boxes survived NMS (NMS_IOU_THRESHOLD). The model is not good enough or something is wrong in code."
        )
        return torch.empty((0, 7), device=boxes.device, dtype=boxes.dtype)

    return torch.cat(final_nms_boxes, dim=0)


# Test as module
# $         python -m utils.nms
def test():
    from configs.config_loader import load_config
    from data.utils.setup_transforms import setup_transforms
    from data.dataset_loader import dataset_loader
    from data.voc_dataset import VOCDataset
    from model.yolov1 import YOLOv1
    from data.utils.bbox_extraction import (
        extract_and_convert_label_bboxes,
        extract_and_convert_pred_bboxes,
    )

    cfg = load_config(
        "config_voc_dataset.yaml", print_configs=False, verify_ask_user=False
    )
    t = setup_transforms(cfg.IMAGE_SIZE)
    d = VOCDataset(
        cfg=cfg,
        which_dataset=cfg.TRAIN_DIR_NAME,
        num_samples=cfg.NUM_TRAIN_SAMPLES,
        transforms=t,
    )

    # === USE THE TRUE LABEL AS TEST ===
    img1, label1 = d.__getitem__(1)
    boxes = torch.stack([label1])
    ext_bboxes = extract_and_convert_label_bboxes(cfg, labels=boxes)

    # === For testing twick the labeled data. ===
    # ext_bboxes[:, 2] = 0.2 # set all pc to be lower than the min threshold
    # adjust pc score of each box
    ext_bboxes[0, 2] = 1.5
    ext_bboxes[1, 2] = 2.1
    ext_bboxes[3, 2] = 0.8

    # adjust the x1, y1, x2, y2 of each box to make them more similar so we can test IoU.
    # ext_bboxes[1, 3] = 119 # make box @ index 1 coords == to box @ index 0 both have the same class ID
    ext_bboxes[1, 3] = 119
    ext_bboxes[1, 4] = 140
    ext_bboxes[1, 5] = 176
    ext_bboxes[1, 6] = 196

    # Append the same box to see it suppressed.
    box_to_repeat = ext_bboxes[0].unsqueeze(0)  # reshape from (7) -> (1, 7)

    ext_bboxes = torch.cat([ext_bboxes, box_to_repeat, box_to_repeat])

    print("All bounding boxes:\n", ext_bboxes, "\n\n")

    # === NMS ===
    print(nms(cfg=cfg, boxes=ext_bboxes))


if __name__ == "__main__":
    test()
