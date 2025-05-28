from torch import Tensor, argmax
from typing import Optional
from argparse import Namespace

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

# Trash classes
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
    config: Namespace,
    true_bboxes: Optional[Tensor] = None,
    pred_bboxes: Optional[Tensor] = None,
):
    """
    Plots the predicted and true bounding boxes over an image.

    Note: the x and y coordinates need to be relative to a single cell, width and height relative to the entire image.

    Parameters
    ----------
        img : PIL.Image
            The image.
        config : Namespace
            Configurations.
        true_bboxes : torch.Tensor
            Optional plot the true bounding boxes. Shape of (S, S, NUM_NODES_PER_CELL). Defaults to None.
        pred_bboxes : torch.Tensor
            Optional plot the pred_bboxes bounding boxes. Shape of (S, S, NUM_NODES_PER_CELL). Defaults to None.
    """

    S = config.S
    # get image size
    img_width, img_height = img.size

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # TODO make this work dry with model output after NMS
    for i in range(S):
        for j in range(S):
            cell = true_bboxes[i, j]
            objectness = cell[18]
            if (
                objectness == 1
            ):  # TODO for pred != 0, and pc can be either cell[18] or cell[23]

                # get the objects name idx
                class_idx = argmax(cell[:18]).item()

                # Grab the first bbox coordinates
                cx_cell, cy_cell = cell[19], cell[20]
                w, h = cell[21], cell[22]

                # Calculate the absolute center relative to the entire image.
                abs_cx = (j + cx_cell) / S
                abs_cy = (i + cy_cell) / S
                # Width and height are already relative to the full image

                # Convert (center x, center y, width, height) to (top-left x, top-left y, width, height)
                x1 = (abs_cx - w / 2) * img_width
                y1 = (abs_cy - h / 2) * img_height
                width = w * img_width
                height = h * img_height

                # Create a rectangle
                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="purple",
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add object name label on-top the bbox
                ax.text(
                    x1,
                    y1 - 5,  # 5 pixels above the box
                    classes[class_idx],
                    fontsize=6,
                    color="purple",
                    verticalalignment="bottom",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.4,
                        edgecolor="none",
                        boxstyle="round,pad=0.2",
                    ),
                )

    # Add legend
    legend_patch = patches.Patch(color="purple", label="True Labels")
    ax.legend(handles=[legend_patch])

    # plt.axis('off')
    plt.show()
