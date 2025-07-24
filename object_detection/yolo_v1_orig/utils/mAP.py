"""Mean Average Precision"""

import torch
from tqdm import tqdm
from collections import Counter

# My modules
from data.utils.bbox_utils import (
    extract_and_convert_pred_bboxes,
    extract_and_convert_label_bboxes,
)
from model.yolov1 import YOLOv1
from data.dataset_loader import dataset_loader
from configs.config_loader import YOLOConfig
from utils.IoU import IoU


def mAP(cfg: YOLOConfig, val_loader: dataset_loader, yolo: YOLOv1):
    """
    Compute Mean Average Precision, it is typically calculated on the validation dataset during training.

    Args:
        cfg: Project configurations.
        val_loader: An instance of the dataset_loader to load the validation dataset.
        yolo: An instance of the Yolo v1 model.

    Return:
        Float: The final mean average precision score.
    """
    print("COMPUTING MEAN AVERAGE PRECISION")

    S, C, B = cfg.S, cfg.C, cfg.B
    mAP_IoU_threshold = 0.5

    # Set the model to evaluation mode.
    yolo.eval()

    # --- 1: Get all predictions and ground truths from the entire validation set ---
    all_pred_boxes = []
    all_true_boxes = []

    # Disable gradient calculations to save memory and computation during inference.
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(
            tqdm(val_loader, leave=True, desc="Validation Batches")
        ):
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            # prettier-ignore x=(batch_size, 3, img_size, img_size) 3 is the number of image channels. y=(batch_size, S, S, CELL_NODES)
            out = yolo(x)  # (b_s, 1470)

            # Reshape output -> # (b_s, S, S, CELL_NODES)
            b_s = x.size(0)  # Batch size not hardcoded for worst-case.
            out = out.view(b_s, S, S, cfg.CELL_NODES)  # (b_s, S, S, CELL_NODES)

            # Extract box data from the current batch and append to master lists.
            # The lists will contain all boxes from all batches.
            pred_boxes_batch = extract_and_convert_pred_bboxes(cfg, pred=out)
            true_boxes_batch = extract_and_convert_label_bboxes(cfg, labels=y)

            all_pred_boxes.extend(pred_boxes_batch)
            all_true_boxes.extend(true_boxes_batch)

    # Set model back to training mode.
    yolo.train()

    # --- 2: Calculate Average Precision (AP) for each class ---
    average_precisions = []  # List storing AP for each class
    epsilon = 1e-6  # For numerical stability

    # Calculate AP for each class individually
    for c in range(C):  # c is the class index
        detections = []  # All predicted boxes for the current class 'c'.
        ground_truths = []  # All ground truth boxes for the current class 'c'.

        # Filter boxes by the current class 'c'
        for detection in all_pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in all_true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # If there are no ground truth boxes for this class, skip it.
        if not ground_truths:
            continue

        # --- Count how many ground truth boxes are in each image for this class ---
        # e.g., amount_bboxes = {0: 3, 1: 2} means image 0 has 3 GTs, image 1 has 2 GTs.
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Convert the counts to zero tensors to track which GTs have been detected.
        # e.g., amount_bboxes = {0: tensor([0, 0, 0]), 1: tensor([0, 0])}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort all detections for this class by confidence score in descending order.
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # --- Core Logic: Match predictions to ground truths ---
        # Iterate through each sorted detection to classify it as TP or FP.
        for detection_idx, detection in enumerate(detections):
            # Get the ground truths that are in the same image as the detection.
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            best_iou = 0
            best_gt_idx = -1

            # Find the best matching ground truth box for the current detection.
            for idx, gt in enumerate(ground_truth_img):
                # Calculate IoU between the current detection and one ground truth
                # Both 'detection' and 'gt' are tensors, so we slice them directly.
                pred_box_tensor = detection[3:]  # pred [x1, y1, x2, y2]
                gt_box_tensor = gt[3:]  # true [x1, y1, x2, y2]
                iou = IoU(pred_box_tensor, gt_box_tensor, box_format="corners")
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Check if the best match is above the IoU threshold.
            if best_iou > mAP_IoU_threshold:
                # Check if this best-matching ground truth has NOT been claimed yet.
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1  # It's a True Positive
                    amount_bboxes[detection[0]][
                        best_gt_idx
                    ] = 1  # Mark this GT as claimed
                else:
                    FP[detection_idx] = 1  # It's a False Positive (duplicate detection)
            else:
                FP[detection_idx] = 1  # It's a False Positive (low IoU)

        # --- Calculate Precision-Recall curve ---
        # Cumulative sums give us the total TPs and FPs at each step.
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # To calculate AP, we integrate under the P-R curve.
        # We add points (0, 1) at the start and (recall_end, 0) at the end.
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Use the trapezoidal rule to calculate the area under the curve.
        average_precisions.append(torch.trapz(precisions, recalls))

    # --- 3: Calculate final mAP ---
    # mAP is the mean of the Average Precisions for all classes.
    mean_ap = sum(average_precisions) / (len(average_precisions) + epsilon)
    print(f"\nVALIDATION mAP: {mean_ap:.4f}")
    return mean_ap


# Test as module:
#    python -m utils.mAP
def test():
    from configs.config_loader import load_config
    from data.utils.setup_transforms import setup_transforms
    from data.dataset_loader import dataset_loader
    from data.voc_dataset import VOCDataset

    cfg = load_config("config_voc_dataset.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)
    yolo = YOLOv1(cfg=cfg, in_channels=3).to(cfg.DEVICE)

    # Load the Validation set.
    val_loader = dataset_loader(
        cfg=cfg,
        which_dataset=cfg.VALIDATION_DIR_NAME,
        num_samples=cfg.NUM_VAL_SAMPLES,
        transforms=t,
        Dataset=VOCDataset,
        # For validation we only one batch.
        batch_size=cfg.VAL_BATCH_SIZE,
    )

    # --- Test mAP() ---
    mAP(cfg, val_loader=val_loader, yolo=yolo)


if __name__ == "__main__":
    test()
