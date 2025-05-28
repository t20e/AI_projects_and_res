"""Non-Max Suppression"""

from .intersection_over_union import intersection_over_union
import torch
from utils.bboxes import extract_bboxes, reconstruct_tensor


def non_max_suppression(pred_bboxes, config):
    """
    Computes NMS by filtering out overlapping bboxes per class.

    Note:
        Non-Maximum Suppression (NMS) is a vital post-processing step in many computer vision tasks, particularly in object detection. It is used to refine the output of object detection models by eliminating redundant bounding boxes and ensuring that each object is detected only once.

    Args:
        pred_bboxes (Tensor): (N, 9) [ i, j, b, class_idx, pc, x, y, w, h]
        config : Namespace configurations.
            Project configurations.

    Variables:
        IOU_threshold (float) : the iou threshold when comparing bounding boxes for NMS.

        min_threshold (float) : the threshold to remove bounding boxes with a low confidence score.
    Returns:
        Tensor : (M, 9) with filtered bboxes.
    """

    DEVICE, IOU_THRESHOLD, MIN_THRESHOLD = (
        config.DEVICE,
        config.IOU_THRESHOLD,
        config.MIN_THRESHOLD,
    )
    
    # --- 1: Filter out low-confidence bboxes; tensors where pc/probability_score (column 4) < min_threshold.
    mask = pred_bboxes[:, 4] >= MIN_THRESHOLD
    bboxes = pred_bboxes[mask]

    # --- 2: Perform NMS
    # store the bboxes that pass IOU
    output = []

    # --- 3: Loop thru the number of unique class_idx
    for cls_idx in bboxes[:, 3].unique():

        # --- 4: Use a mask so we can put the boxes with the same class_idx into a tensor.
        class_mask = bboxes[:, 3] == cls_idx
        classes_bboxes = bboxes[
            class_mask
        ]  # classes_bboxes -> is a tensor that contains boxes with the same class_idx!
        keep = []  # Stores boxes to keep

        # --- 5: Queue -> Loop thru the boxes of the same class_idx.
        while len(classes_bboxes) > 0:
            """This queue works like so
            1. classes_bboxes = [box1, box2, box3, etc..] all of the same class_idx
            2. chosen_box = box1
            3. box1 is add to keep
            3. box1 is compared with all the other boxes vectorized
            4. if any box# overlap too much/doesn't pass IOU with box1, then those boxes are removed from queue list.
            5. then we loop back up, and chosen_box is the next box in queue that didn't overlap with box1 etc..
            """
            # --- 6: Get the first box, which will always be the box with the highest pc for every class.
            chosen_bbox = classes_bboxes[0]
            keep.append(chosen_bbox)  # since it has the highest pc, safe to keep it

            # Handle final case
            if len(classes_bboxes) == 1:
                break

            # Pop the chosen_bbox, so we get a tensor with the rest of the box of the same class_idx.
            rest = classes_bboxes[1:]

            #  --- 7: Compute IOU
            iou = intersection_over_union(chosen_bbox=chosen_bbox, rest_bbox=rest)

            # --- 8: Remove overlapping boxes
            classes_bboxes = rest[iou < IOU_THRESHOLD]

        # Add valid boxes to output
        output.extend(keep)
    if len(output) == 0:
        print("OUTPUT LENGTH EMPTY")
    return torch.stack(output, dim=0)


# test function

# from types import SimpleNamespace

# # Simulated config
# config = SimpleNamespace(DEVICE="cpu", IOU_THRESHOLD=0.5)

# # Create test tensor [i, j, b, class_idx, pc, x, y, w, h]
# test_tensor = torch.tensor(
#     [
#         [0, 0, 0, 1, 0.95, 0.5, 0.5, 0.4, 0.4],  # keep
#         [0, 0, 1, 1, 0.85, 0.52, 0.52, 0.4, 0.4],  # suppress (overlaps)
#         [0, 0, 0, 1, 0.30, 0.9, 0.9, 0.3, 0.3],  # keep (low overlap)
#         [0, 1, 1, 2, 0.88, 0.5, 0.5, 0.2, 0.2],  # keep (diff class)
#         [0, 1, 0, 2, 0.70, 0.51, 0.51, 0.2, 0.2],  # suppress (same class + overlaps)
#     ]
# )

# # Run
# filtered = vectorized_nms(test_tensor, config)

# # Print results
# print("Filtered BBoxes:")
# print(filtered)
