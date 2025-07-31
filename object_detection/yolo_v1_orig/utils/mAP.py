"""Mean Average Precision"""

import torch
from tqdm import tqdm

# My modules
from data.utils.bbox_utils import (
    extract_and_convert_pred_bboxes,
    extract_and_convert_label_bboxes,
)
from utils.nms import nms
from model.yolov1 import YOLOv1
from data.dataset_loader import dataset_loader
from configs.config_loader import YOLOConfig
from utils.IoU import IoU_one_to_many_mapping


def mAP(cfg: YOLOConfig, val_loader: dataset_loader, yolo: YOLOv1):
    """
    Compute Mean Average Precision, which is typically computed on the validation dataset during training.

    Args:
        cfg: Project configurations.
        val_loader: An instance of the dataset_loader that loads the validation dataset.
        yolo: An instance of the YOLO v1 model.

    Return:
        Float: The final mean average precision score.
    """
    print("\nCOMPUTING MEAN AVERAGE PRECISION")

    S, C, B = cfg.S, cfg.C, cfg.B
    epsilon = 1e-6  # For numerical stability

    # Define a threshold that if a predicted box and a target box over-lap more than this threshold than that predicted box is a good prediction.
    mAP_IoU_threshold = 0.5  # 0.5 is common

    # --- 1: Get all predictions and ground truths from all the batches in the entire validation set ---
    all_pred_boxes_list = []  # List to collect tensors
    all_true_boxes_list = []  # List to collect tensors

    current_img_offset = 0  # To assign unique image indices across batches.

    # Set the model to evaluation mode.
    yolo.eval()

    # Disable gradient calculations to save memory, i.e, not training the model.
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(
            tqdm(val_loader, leave=True, desc="Validation Batches")
        ):
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            # prettier-ignore x=(batch_size, 3, img_size, img_size) 3 is the number of image channels. y=(batch_size, S, S, CELL_NODES)
            out = yolo(x)  # (b_s, 1470)
            b_s = x.size(0)  # Batch size not hardcoded for worst-case.

            # Reshape output -> (b_s, S, S, CELL_NODES)
            out = out.view(b_s, S, S, cfg.CELL_NODES)  # (b_s, S, S, CELL_NODES)

            # Extract box data from the current batch and append to master lists.
            pred_boxes_batch = extract_and_convert_pred_bboxes(
                cfg, pred=out, img_offset=current_img_offset
            )
            true_boxes_batch = extract_and_convert_label_bboxes(
                cfg, labels=y, img_offset=current_img_offset
            )
            if pred_boxes_batch.numel() > 0:  # Check if tensor is not empty
                all_pred_boxes_list.append(pred_boxes_batch)
            if true_boxes_batch.numel() > 0:  # Check if tensor is not empty
                all_true_boxes_list.append(true_boxes_batch)

            # Add an offset to distinguish (batch_1's image_1) from (batch_2's image_1)
            current_img_offset += b_s

    # Set model back to training mode.
    yolo.train()

    # --- 2: Convert lists to tensors and Apply NMS to the predictions ---
    if not all_true_boxes_list:
        print("\nNo ground truth boxes found across all batches. mAP is 0.")
        return 0.0
    if not all_pred_boxes_list:
        print("\nNo predictions found across all batches. mAP is 0.")
        return 0.0

    all_pred_boxes_tensor = torch.cat(all_pred_boxes_list, dim=0)  # (num_boxes, 7)
    
    # --> Apply NMS to the predictions to remove redundant bboxes.
    NMS_all_pred_boxes_tensor = nms(cfg, all_pred_boxes_tensor)
    # Sort the NMS-filtered predictions by confidence descending, crucial for mAP.
    if NMS_all_pred_boxes_tensor.numel() == 0:
        print("\nNo bounding boxes survived NMS. mAP is 0.")
        return 0.0

    NMS_all_pred_boxes_tensor = NMS_all_pred_boxes_tensor[
        NMS_all_pred_boxes_tensor[:, 2].argsort(descending=True)
    ]
    # Ground truths don't need NMS
    all_true_boxes_tensor = torch.cat(all_true_boxes_list, dim=0)  # (num_boxes, 7)

    # --- 3: Calculate Average Precision (AP) for each class ---
    average_precisions = []  # List storing AP for each class.

    # Calculate AP for each class individually
    for c in range(C):
        # Note: detection is when the model predicted a bounding box, doesn't mean that that bbox is correct though.
        # 3.1: Filter detections and ground truths for the current class
        class_detections = NMS_all_pred_boxes_tensor[
            NMS_all_pred_boxes_tensor[:, 1] == c
        ]
        class_ground_truths = all_true_boxes_tensor[all_true_boxes_tensor[:, 1] == c]
        # shapes: (num_boxes with the same class, 7)

        if len(class_ground_truths) == 0:
            continue  # Skip classes with no ground truth objects

        # 3.2: Track which ground truths have been matched (for each image)
        amount_bboxes_seen = {}
        unique_img_indices = torch.unique(class_ground_truths[:, 0]).to(
            torch.int64
        )  # Ensure int for indexing
        for img_idx_tensor in unique_img_indices:
            img_idx = img_idx_tensor.item()
            num_gt_in_img = (class_ground_truths[:, 0] == img_idx).sum().item()
            amount_bboxes_seen[img_idx] = torch.zeros(
                num_gt_in_img, dtype=torch.bool, device=cfg.DEVICE
            )

        # TP = True positive | FP = False positive
        TP = torch.zeros(len(class_detections), device=cfg.DEVICE)
        FP = torch.zeros(len(class_detections), device=cfg.DEVICE)
        total_true_bboxes = len(class_ground_truths)

        if len(class_detections) == 0:
            # If no prediction was made for this class append class (AP=0.0), continue.
            average_precisions.append(torch.tensor(0.0, device=cfg.DEVICE))
            continue

        # 3.3: Iterate through each individual predicted bounding box that belongs to the current class c, regardless of which image it came from.
        for det_idx, detection in enumerate(class_detections):
            img_idx = int(detection[0].item())

            # Get ground truths only for the current image and class
            gts_in_current_img_mask = class_ground_truths[:, 0] == img_idx
            gts_box_in_img = class_ground_truths[gts_in_current_img_mask]

            if len(gts_box_in_img) == 0:
                FP[det_idx] = 1  # No GTs in this image, so detection is FP
                continue

            # Calculate IoU between current detection and all GTs in the same image
            # For mAP, you need to calculate the IoU of one predict bbox against all ground truths in the same image.
            ious = IoU_one_to_many_mapping(
                detection[3:].unsqueeze(0),
                gts_box_in_img[:, 3:],
                box_format="corners",
            )  # (1, num_gt_in_img)
            best_iou, best_gt_in_img_idx = torch.max(ious, dim=1)
            best_iou = best_iou.item()
            best_gt_in_img_idx = best_gt_in_img_idx.item()

            # Check if the best match is above the IoU threshold.
            if best_iou > mAP_IoU_threshold:
                # Check if this best-matching ground truth has NOT been claimed yet.
                if not amount_bboxes_seen[img_idx][best_gt_in_img_idx]:
                    TP[det_idx] = 1  # It's a True Positive
                    amount_bboxes_seen[img_idx][
                        best_gt_in_img_idx
                    ] = True  # Mark this Ground Truth as claimed.
                else:
                    FP[det_idx] = 1  # It's a False Positive (duplicate detection).
            else:
                FP[det_idx] = 1  # It's a False Positive (low IoU).

        # 3.4: Calculate Precision-Recall curve
        # Cumulative sums give us the total TPs and FPs at each step.
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # Handle division by zero for recalls and precisions
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # Add (0,1) and (recall_end, 0) for area under curve calculation
        precisions = torch.cat((torch.tensor([1.0], device=cfg.DEVICE), precisions))
        recalls = torch.cat((torch.tensor([0.0], device=cfg.DEVICE), recalls))

        # Use torch.trapz for area under the curve
        average_precisions.append(torch.trapz(precisions, recalls))

    # --- 4: Compute Mean ---
    #   mAP is the mean of the Average Precisions for all classes.
    if not average_precisions:
        print("\nNo classes with ground truth boxes found. mAP is 0.")
        return 0.0

    mean_ap = sum(average_precisions) / (len(average_precisions) + epsilon)
    print(f"\nVALIDATION mAP: {mean_ap.item():.4f}")
    return mean_ap.item()


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
    print(mAP(cfg, val_loader=val_loader, yolo=yolo))


if __name__ == "__main__":
    test()
