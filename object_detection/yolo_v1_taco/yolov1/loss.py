import torch
import torch.nn as nn
from argparse import Namespace

from utils.intersection_over_union import intersection_over_union
from utils.bboxes import extract_bboxes

class YoloLoss(nn.Module):
    def __init__(self, config: Namespace):
        """
        Yolo v1 loss function.

        Args:
            config (argparse.Namespace): project configurations.

        """
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.config = config
        self.lambda_noobj = 0.5  # no object exists.
        self.lambda_coord = 5  # constant from the paper.