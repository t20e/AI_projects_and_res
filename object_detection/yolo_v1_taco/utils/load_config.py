""" NOTE: for future projects use a dataclass to store configurations instead of a Namespace.
    from dataclasses import dataclass
    import yaml
    @dataclass
    class Config:
"""


import yaml
from argparse import Namespace
import torch

def load_config():
    """Returns project configurations"""
    config = Namespace(**yaml.safe_load(open("./config.yaml")))

    # NOTE: NUM_NODES_PER_CELL and NUM_NODES_PER_IMG could have been named better like "D" or "M" idk
    # The total number of nodes that a single cell has in a label for one image, which would be the size -> [*classes, pc_1, bbox1_x_y_w_h, pc_2, bbox2_x_y_w_h]. If S=7 C=18 B=2 --> 28 nodes.
    config.NUM_NODES_PER_CELL = config.C + 5 * config.B

    # # The total number of nodes that each label has for one image. If S=7 C=18 B=2 --> 7 * 7 * (18 + 2 * 5) = 1,372 | 7x7=49 -> 49*28 = 1,372 | the * 5 is for the second bbox in the cell -> pc_2, x, y, w, h
    config.NUM_NODES_PER_IMG = config.S * config.S * (config.C + config.B * 5)

    # set DEVICE to torch.device
    config.DEVICE = torch.device(config.DEVICE)

    # Convert "2e-5" to a float()
    config.LEARNING_RATE = float(config.LEARNING_RATE)

    return config

# load_config()