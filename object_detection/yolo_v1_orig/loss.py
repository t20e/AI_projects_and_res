import torch
import torch.nn as nn
from configs.config_loader import YOLOConfig


class YOLOLoss(nn.Module):
    def __init__(self, cfg: YOLOConfig):
        """
        YOLO v1 Loss function.

        Args:
            cfg: project configurations.

        """
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(
            reduction="sum"
        )  #  CrossEntropy/Softmax losses normally work better than MSE.
        self.config = cfg
        # The lambdas from paper.
        self.LAMBDA_COORD = 5  # λ_coord
        self.LAMBDA_NOOBJ = 0.5  # λ_noobj

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        """
        Compute YOLOv1 loss.

        Note:
            Where u see example (n_s, S, S, C) or (64, 7, 7) etc.. in the comments below is the tensor's shape.
        Args:
            pred (tensor): shape (batch_size, S, S, num_nodes_per_cell)
            label (tensor): shape (batch_size, S, S, num_nodes_per_cell)

        Returns:
            Loss (float)
        """
        # TODO TODO TODO
