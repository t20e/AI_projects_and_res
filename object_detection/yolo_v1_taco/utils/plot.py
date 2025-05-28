from torch import Tensor, argmax
from typing import Optional
from argparse import Namespace

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from utils.bboxes import convert_yolo_to_corners

# Object classes, retrieve by index
classes = [
    "Aluminum foil",
    "Bottle",
    "Bottle cap",
    "Broken glass",
    "Can",
    "Carton",
    "Cigarette",
    "Cup",
    "Lid",
    "Other litter",
    "Other plastic",
    "Paper",
    "Plastic bag & wrapper",
    "Plastic container",
    "Pop tab",
    "Straw",
    "Styrofoam piece",
    "Unlabeled litter",
]


def plot_bboxes(
    img: Image,
    label_bboxes: Optional[Tensor] = None,
    pred_bboxes: Optional[Tensor] = None,
    S: int=7
):
    
    """Plot the predicted and true bounding boxes over an image.
    
    Note: bboxes must be passed in in mid-point format.

    Args:
        img (PIL.Image): the image to plot over.
        label_bboxes (Tensor): Shape(N,9) -> [ i, j, b, class_idx, pc, x, y, w, h]
        pred_bboxes (Tensor): Shape(N,9) -> [ i, j, b, class_idx, pc, x, y, w, h]
    
    """
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    legend_handles = []

    if pred_bboxes is not None:
        pred_legend = draw_box(ax=ax, bboxes=pred_bboxes, S=S, img_s=img.size[0], color="magenta", name="Pred")
        legend_handles.append(pred_legend)
    if label_bboxes is not None:
        label_legend = draw_box(ax=ax, bboxes=label_bboxes, S=S, img_s=img.size[0], color="blue", name="Label")
        legend_handles.append(label_legend)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    # plt.axis("off")
    plt.show


def draw_box(ax, bboxes, S, img_s, color, name):
    """Helper function to plot the boxes"""
    # Convert from mid-point to corner-points
    coords = convert_yolo_to_corners(bboxes, S, img_s)

    for i, (x1, y1, x2, y2) in enumerate(coords.tolist()):
        # get info
        class_idx = int(bboxes[i, 3])
        pc = float(bboxes[i, 4])

        # add box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add object name label on-top the bbox
        ax.text(
            x1,
            y1 - 5,  # 5 pixels above the box
            f"{classes[class_idx]} ({pc:.2f})",
            fontsize=6,
            color=color,
            verticalalignment="bottom",
            bbox=dict(
                facecolor="white",
                alpha=0.4,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
        )

    # Add to legend
    return patches.Patch(color=color, label=name)




