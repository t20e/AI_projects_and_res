"""
Bounding box conversion (Mid-points <-> corner-points) and (absolute pixel values <-> normalized values).

NOTE: most of these functions could be applied in the functions that will call it, however, to make the code clearer and less chances of bugs (like mistakenly passing the wrong format thru IoU, NMS, etc...), I decided to make a function for each.
"""

import torch


def convert_x_y_rel_cell_to_rel_image(
    x_rel_cell: torch.Tensor,
    y_rel_cell: torch.Tensor,
    i_idxs: torch.Tensor,
    j_idxs: torch.Tensor,
    S: int = 7,
):
    """
    Convert (x, y) box midpoints from being relative to a single cell, to be relative to the entire image, and still be in midpoint format. Note: w, h are relative to the entire image.

    Args:
        x_rel_cell (torch.Tensor): x-coordinate relative to the cell (0-1).
        y_rel_cell (torch.Tensor): y-coordinate relative to the cell (0-1).
        i_idxs (torch.Tensor): Row indices of the grid cell.
        j_idxs (torch.Tensor): Column indices of the grid cell.
        S (int): The grid size (e.g., 7 for a 7x7 grid).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The x and y midpoints relative to the image (0-1).
    """
    x_mid_norm = (x_rel_cell + j_idxs) / S
    y_mid_norm = (y_rel_cell + i_idxs) / S
    return x_mid_norm, y_mid_norm


def convert_mid_to_corner(
    x_mid: torch.Tensor, y_mid: torch.Tensor, w: torch.Tensor, h: torch.Tensor
):
    """
    Convert bounding box coordinates from midpoint (x, y, w, h) to corner-points (x1, y1, x2, y2).

    Args:
        x_mid (torch.Tensor): Midpoint x-coordinate normalized to the entire image.
        y_mid (torch.Tensor): Midpoint y-coordinate normalized to the entire image.
        w (torch.Tensor): Bounding box width normalized to the entire image.
        h (torch.Tensor): Bounding box height normalized to the entire image.

            Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The four corner coordinates.
    """
    x1 = x_mid - w / 2
    y1 = y_mid - h / 2
    x2 = x_mid + w / 2
    y2 = y_mid + h / 2
    return x1, y1, x2, y2



def convert_norms_to_abs(coords: torch.Tensor, image_size: int):
    """
    Convert normalized coordinates (0-1) to absolute pixel values e.g: (100pixels x 100pixels). The coords can be passed in as either corner-points or mid-points.

    Args:
        coords (torch.Tensor): Example: normalized x₁ tensor.
        image_size (int): The size of the image (e.g., 448 for a 448x448 image).

    Returns:
        torch.Tensor: The coordinates in absolute pixel values.
    """
    return coords * image_size


def convert_abs_to_norm(coords: torch.Tensor, image_size: int):
    """
    Convert absolute pixel values to normalized coordinates (0-1). The coords can be passed in as either corner-points or mid-points.

    Args:
        coords (torch.Tensor): Example: x₁ tensor of absolute pixel values.
        image_size (int): The size of the image (e.g., 448 for a 448x448 image).

    Returns:
        torch.Tensor: The coordinates in normalized values (0-1).
    """
    return coords / image_size
