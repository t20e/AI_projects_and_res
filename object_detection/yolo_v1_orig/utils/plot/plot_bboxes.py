"""Plot bboxes on images"""

from configs.config_loader import YOLOConfig
from PIL import Image, ImageDraw
from torch import Tensor
from IPython.display import display  # for jupyter notebook to display images in-line


def plot_bbox(cfg: YOLOConfig, bboxes: Tensor, image: Image):
    """
    Plot bounding boxes on images

    Args:
        cfg: Project configurations
        bboxes (tensor): Shape (N, 7) where N is the number of boxes and 7 is [image_idx, class_idx, pc_score, x1, y1, x2, y2]
            - Format: Absolute corner-points.
        image (PIL.Image): the image to plot bboxes over.
    """
    #  Define a list of 20 colors one for each class.
    colors = [ 
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 128), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
        (0, 128, 0), (0, 128, 128), (0, 0, 128), (255, 165, 0),
        (255, 215, 0), (173, 255, 47), (255, 20, 147), (0, 250, 154), (70, 130, 180)
    ] # fmt: skip

    draw = ImageDraw.Draw(image)

    # Decided not to include image index as it would cause confusion, the box from the extractors are not in sync with dataset dataframe.
    ## Add image index
    ## draw.rectangle([ 0, 0, 110, 12 ], fill=(0, 0, 0), ) # fmt: skip
    ## img_idx = round(bboxes[0, 0:1].item())
    ## draw.text((5, 0), f"Image Index: {img_idx}", fill=(255, 255, 255))

    for box in bboxes:
        img_idx, cls_idx, pc_score, x1, y1, x2, y2 = box
        cls_idx = int(cls_idx)

        # Choose color based on class_id
        color = colors[cls_idx % len(colors)]

        # --- Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # --- Draw the label with class ID
        label = f"Class: {cfg.CLASS_NAMES[cls_idx]} pc: {round(pc_score.item(), 3)}"

        text_size = draw.textbbox((0, 0), label)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        # --- Position the label
        if y1 - text_height >= 0:
            text_origin = (x1, y1 - text_height)
        else:
            text_origin = (x1, y1 + 1)
        # --- Draw a filled rectangle for the text background
        draw.rectangle(
            [
                text_origin[0],
                text_origin[1],
                text_origin[0] + text_width,
                text_origin[1] + text_height,
            ],
            fill=color,
        )
        # --- Draw the text
        draw.text(text_origin, label, fill=(0, 0, 0))

    # image.show()
    display(image)  # for jupyter notebooks
