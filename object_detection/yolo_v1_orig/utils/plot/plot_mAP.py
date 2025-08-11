import matplotlib.pyplot as plt


def plot_pr_curve(
    precisions,
    recalls,
    all_cls_names,
    cls_idx="",
):
    """
    Plots the Precision-Recall curve.

    Args:
        precisions (torch.Tensor): A tensor of precision values.
        recalls (torch.Tensor): A tensor of recall values.
        all_cls_names: List of all the class names.
        cls_idx (str): The name of the class for the plot title.
    """
    # Detach tensors from GPU if necessary and convert to numpy
    precisions_np = precisions.cpu().numpy()
    recalls_np = recalls.cpu().numpy()

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        recalls_np,
        precisions_np,
        marker="o",
        linestyle="--",
        color="b",
        label="PR Curve",
    )

    # Add labels and title
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve for Class -> ID: {cls_idx}, {all_cls_names[cls_idx]}"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.show()


# Test as module:
# $         python -m utils.plot.plot_mAP
def test_pr_curve_plotting():
    import torch

    # Simulate a scenario with 5 detections and 3 total ground truth boxes
    # The detections are sorted by confidence, and their TP/FP status is assigned based on IoU matching.

    # Let's say there are 3 total ground truth boxes for this class.
    total_true_bboxes = 3

    # Detections sorted by confidence:
    # Detection 1: Correct (TP)
    # Detection 2: Incorrect (FP)
    # Detection 3: Correct (TP)
    # Detection 4: Incorrect (FP)
    # Detection 5: Correct (TP)
    TP_status = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
    FP_status = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)

    # Cumulative sums
    TP_cumsum = torch.cumsum(TP_status, dim=0)
    FP_cumsum = torch.cumsum(FP_status, dim=0)

    # Calculate precision and recall
    epsilon = 1e-6
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)

    # Add the (0, 1) point to the curve
    precisions = torch.cat((torch.tensor([1.0]), precisions))
    recalls = torch.cat((torch.tensor([0.0]), recalls))

    # All class names list
    all_cls_names = ["class_0", "class_1", "class_2", "class_3", "class_4"]
    cls_idx = 2  # The class index we are plotting

    # Call the plotting function with the generated data
    print(
        f"Generating Precision-Recall curve for Class -> ID: {cls_idx}, {all_cls_names[cls_idx]}"
    )
    plot_pr_curve(
        precisions=precisions,
        recalls=recalls,
        all_cls_names=all_cls_names,
        cls_idx=cls_idx,
    )
    print("Plotting complete. Please close the plot window to continue.")


if __name__ == "__main__":
    test_pr_curve_plotting()
