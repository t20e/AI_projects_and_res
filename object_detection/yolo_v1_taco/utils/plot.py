from torch import Tensor
from typing import Optional
from argparse import Namespace

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def plot_bboxes(img:Image, config:Namespace, true_bboxes:Optional[Tensor] = None, out_bboxes:Optional[Tensor]=None):
    """
    Plots the predicted and true bounding boxes over an image.

    Parameters
    ----------
        img : PIL.Image
            The image.
        config : Namespace
            Configurations.
        true_bboxes : torch.Tensor
            Optional plot the true bounding boxes. Shape of (# of bboxes, NUM_NODES_PER_CELL). Defaults to None.
        out_bboxes : torch.Tensor
            Optional plot the out_bboxes bounding boxes. Shape of (# of bboxes, NUM_NODES_PER_CELL). Defaults to None.
    """



    # Plotting
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Draw each bounding box
    img_width, img_height = img.size

    for bbox in true_bboxes:
        if bbox[18] == 1:  # Only for valid bboxes
            cx, cy, w, h = bbox[19], bbox[20], bbox[21], bbox[22]

            # Convert from relative center format to top-left corner format
            x1 = (cx - w / 2) * img_width
            y1 = (cy - h / 2) * img_height
            width = w * img_width
            height = h * img_height

            # Draw the box
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='purple',
                facecolor='none'
            )
            ax.add_patch(rect)

    # Create a custom legend
    ax.legend(handles=[patches.Patch(color='purple', label='TRUE labels')])

    plt.axis('on')  # Optional: remove axes
    plt.show()

