"""Utils for Tensor bounding boxes extract, etc.."""

import torch
from argparse import Namespace


# Extract bounding boxes
# NOTE: Do not implement this function before the loss_fn(). Use this function before plot_bboxes(), and non_max_suppression().
def extract_bboxes(t: torch.Tensor, config: Namespace):
    """
    Extract bounding boxes from a single image's (predicted or labeled) tensor, converting them into a flat (N, 9) representation, and finally sorting based on their probability score (pc) in descending order.

    Note:
        - N = S * S * B or total number of bboxes per image.
        - Nested 9 nodes looks like -> [i, j, b, class_idx, pc, x, y, w, h], sorted by pc descending for each image.
        - i,j is the bboxes cell location in the grid.
        - b is if the bbox is either bbox1 or bbox2 in its cell. Its value is either 0 or 1.
        - pc is the probability score that an object exists in that cell.
        - class_idx is a value between 0 and 17 that indicates which object the cell predicts to be present. It’s obtained by applying argmax() to the class object probability scores at indices 0 through 17. Note object probability scores here is not the same as pc.
        - The return tensor will have bboxes sorted by pc (descending).

    Args:
        t (tensor) : Shape (S, S, NUM_NODES_PER_CELL)

    Returns:
        (tensor) : shape (S * S * B , 9), Sorted by bboxes with the highest pc at the beginning. [[ i, j, b, class_idx, pc, x, y, w, h]] -> num nodes = 9.
    """
    S, B, C, DEVICE, NUM_NODES_PER_CELL = (
        config.S,
        config.B,
        config.C,
        config.DEVICE,
        config.NUM_NODES_PER_CELL,
    )

    # === 1: Create new tensors to store class probs, first bbox and second bbox from every cell across the batch.
    class_probs = t[..., :C]  # ( S, S, C)

    bbox_1 = t[..., C : C + 5]  # ( S, S, 5) #pc1, x1, y1, w1, h1
    bbox_2 = t[..., C + 5 : NUM_NODES_PER_CELL]  # ( S, S, 5)
    bboxes = torch.stack([bbox_1, bbox_2], dim=2)  # shape: ( S, S, 2, 5)

    #   Get the highest predicted object from indexes 0-17. Store as index.
    class_idx = class_probs.argmax(dim=-1)  # shape: ( S, S)

    # === 2: Create grid cell indice mapping tensor for i, j, and b coords.
    # Note: (i,j) -> i = row_indices and j = col_indices
    row_indices, col_indices = torch.meshgrid(
        torch.arange(S, device=DEVICE), torch.arange(S, device=DEVICE), indexing="ij"
    )  # (S, S)

    #       Reshape and expand to include the number bounding boxes per cell.
    row_indices = row_indices.unsqueeze(-1).expand(
        -1, -1, B
    )  # from (S, S) -> (S, S, 2)
    col_indices = col_indices.unsqueeze(-1).expand(
        -1, -1, B
    )  # from (S, S) -> (S, S, 2)
    box_indices = (
        torch.arange(B, device=DEVICE).view(1, 1, B).expand(S, S, B)
    )  # (S, S, 2)

    # === 3: Expand class_idx tensor to match (S, S, 2)
    cls_indices = class_idx.unsqueeze(-1).expand(
        -1, -1, B
    )  # (7, 7) -> (7, 7, 2)

    # === 4: Stack metadata
    metadata = torch.stack(
        [row_indices, col_indices, box_indices], dim=-1
    )  # (7, 7, 2, 3)

    cls_indices = cls_indices.unsqueeze(-1)  # ( 7, 7, 2) -> ( 7, 7, 2, 1)

    #       Concatenate: (S, S , 2, 9) 9 = [i, j, b, class_idx, pc, x, y, w, h]
    full = torch.cat(
        [metadata.float(), cls_indices.float(), bboxes], dim=-1
    )  # (7, 7, 2, 9)

    #      Reshape full into flat form: ( S, S, 2, 9) -> ( N, 9) where N=S*S*2 or total num bboxes per image.
    full = full.view( -1, 9)

    # === 5: Sort by pc (column 4)
    sorted_indices = full[:, 4].argsort(
        descending=True
    )

    return full.index_select(
        0, sorted_indices
    )


def convert_yolo_to_corners(bboxes, S, img_s):
    """Convert from yolo mid-points x, y, w, h to corner-points x1, y1, x2, y2 this is done for plotting.

    Args:
        t (Tensor): Shape(N, 9) -> [ i, j, b, class_idx, pc, x, y, w, h]
    Returns:
        Tensor: Shape(N, 4) -> [x1, y1, x2, y2]
    """
    cell_size = img_s / S  # for 448 images -> 64px per cell

    # Extract Values
    i, j = bboxes[:, 0], bboxes[:, 1]
    x, y, w, h = bboxes[:, 5], bboxes[:, 6], bboxes[:, 7], bboxes[:, 8]

    # Convert x, y from cell-relative to image-relative
    abs_x = (j + x) * cell_size
    abs_y = (i + y) * cell_size

    abs_w = w * img_s
    abs_h = h * img_s

    # convert to corner-point format
    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


# NOTE: The reconstruct was not necessary.
def reconstruct_tensor(bboxes: torch.Tensor, config):
    """
    Reconstruct the filtered bboxes (N, 9) that passed IOU back into a shape of (S, S, NUM_NODES_PER_CELL), the other values will be zero-ed out.

    Args:
        bboxes (Tensor): Shape (N, 9) -> [i, j, b, class_idx, pc, x, y, w, h]
    Returns:
        Tensor: Shape (S, S, NUM_NODES_PER_CELL).

    """
    S, B, C, DEVICE, NUM_NODES_PER_CELL = (
        config.S,
        config.B,
        config.C,
        config.DEVICE,
        config.NUM_NODES_PER_CELL,
    )

    # Create the tensor, default value Zeros
    out = torch.zeros((S, S, NUM_NODES_PER_CELL), device=DEVICE)

    for box in bboxes:
        # extract the values
        i, j, b, class_idx, pc, x, y, w, h = (
            box.long()[:3].tolist() + box[3:].tolist()
        )  # I could've done = box but that could cause indexing issues.

        # Set the class probability index -> class_idx = 1
        out[i, j, int(class_idx)] = 1.0

        # Set box info
        box_offset = (
            C + int(b) * 5
        )  # offset the placement of the box for whether it belongs in box1 or box2 of that cell.

    return out
