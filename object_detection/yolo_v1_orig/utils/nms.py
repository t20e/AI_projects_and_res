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

    # ==> 1. Filter out low-confidence bounding boxes globally
    keep_mask_conf = boxes[:, 2] > NMS_MIN_THRESHOLD
    filtered_boxes = boxes[keep_mask_conf]

    if filtered_boxes.numel() == 0:
        # Return an empty tensor if no boxes survive min-threshold
        print(
            "No bounding boxes has a probability score (pc) > NMS_MIN_THRESHOLD. The model is not good enough or something is wrong in code."
        )
        return torch.empty((0, 7), device=boxes.device, dtype=boxes.dtype)

    final_nms_boxes = []

    # ==> 2. Iterate through unique image IDs from entire dataset.
    # NMS fundamentally needs to be applied per image and per class.
    for img_idx in filtered_boxes[:, 0].unique():  # The image_idx is at index 0
        # Get all bounding boxes for the current image.
        img_mask = filtered_boxes[:, 0] == img_idx
        img_boxes = filtered_boxes[img_mask]  # Shape: (NumBoxesInImg, 7)

        # 2.1 Iterate through each class IDs of bboxes in this image.
        for class_idx in img_boxes[:, 1].unique():  # The class_idx is at index 1
            # Get all boxes for the current class in the current image
            class_mask = img_boxes[:, 1] == class_idx
            class_img_boxes = img_boxes[class_mask]  # Shape: (NumBoxesInImgAndClass, 7)

            # Extract box coordinates (x1, y1, x2, y2) and scores
            box_coords = class_img_boxes[:, 3:7]  # (NumBoxesInImgAndClass, 4)
            box_scores = class_img_boxes[:, 2]
            #    (NumBoxesInImgAndClass,) (confidence score is at index 2)

            # --- NMS Logic implementation ---
            #   Sort boxes by score in descending order (highest confidence first). This is crucial for the greedy NMS algorithm
            scores_sorted, order = box_scores.sort(descending=True)
            sorted_box_coords = box_coords[order]

            #   Keep track of which boxes to retain.
            keep_indices = torch.ones(
                len(sorted_box_coords), dtype=torch.bool, device=boxes.device
            )
            for i in range(len(sorted_box_coords)):
                if not keep_indices[i]:  # If this box has already been suppressed, skip
                    continue
                # Get the current box (the one with highest remaining confidence score).
                current_box = sorted_box_coords[i].unsqueeze(
                    0
                )  # Shape (1, 4) for IoU_for_mAP

                # Compare it against all *subsequent* boxes in the sorted list that are not yet suppressed
                # Only check boxes that are candidates for suppression (i.e., not already marked to be kept=False)
                # And ensure we don't compare a box with itself (j > i)
                candidate_indices = torch.arange(
                    i + 1, len(sorted_box_coords), device=boxes.device
                )
                candidate_mask = keep_indices[
                    candidate_indices
                ]  # Only consider active candidates

                if candidate_mask.sum() == 0:
                    # No more candidates to compare against
                    break

                # Get the coordinates of active candidates
                candidate_boxes = sorted_box_coords[
                    candidate_indices[candidate_mask]
                ]  # Shape (NumActiveCandidates, 4)

                # Calculate IoU between current_box and all active candidates. ( current_box=(1,4), candidate_boxes=(M,4) ) -> (1, M)
                ious = IoU_one_to_many_mapping(
                    current_box, candidate_boxes, box_format="corners"
                )

                # Find candidates that overlap too much with the current box.
                overlapping_candidates_mask = (ious > NMS_IOU_THRESHOLD).squeeze(
                    0
                )  # (M,) bool

                # Suppress (mark for removal) these overlapping candidates.
                # A more robust way: iterate through 'candidate_indices' and update 'keep_indices' directly.
                # Iterate through the indices of candidates that are currently active
                # and check if their IoU with current_box is too high.
                active_candidate_global_indices = order[
                    candidate_indices[candidate_mask]
                ]

                # For each active candidate that overlaps too much, set its corresponding keep_index to False
                for k, should_suppress in enumerate(overlapping_candidates_mask):
                    if should_suppress:
                        # Find the original index of this suppressed box
                        original_index_of_suppressed_box = candidate_indices[
                            candidate_mask
                        ][k].item()
                        keep_indices[original_index_of_suppressed_box] = False

            # Filter the class_img_boxes based on the final keep_indices
            # Re-map keep_indices from sorted order back to original order of class_img_boxes
            # The `order` tensor gives us the mapping from original to sorted.
            # We need to create a `keep_mask_original_order` that applies to `class_img_boxes`.

            # Create a boolean mask in the original order of `class_img_boxes`
            keep_mask_original_order = torch.zeros(
                len(class_img_boxes), dtype=torch.bool, device=boxes.device
            )
            # Populate it using the `order` and `keep_indices` from the sorted list
            keep_mask_original_order[order[keep_indices]] = True

            # Append the boxes that survived NMS for this image and class
            final_nms_boxes.append(class_img_boxes[keep_mask_original_order])

    if not final_nms_boxes:
        # Return an empty tensor if no boxes survive NMS
        print(
            "No bounding boxes survived NMS (NMS_IOU_THRESHOLD). The model is not good enough or something is wrong in code."
        )
        return torch.empty((0, 7), device=boxes.device, dtype=boxes.dtype)

    # Concatenate all survived boxes
    return torch.cat(final_nms_boxes, dim=0)
