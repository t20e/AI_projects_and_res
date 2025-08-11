"""YOLO v1 Model Architecture"""

import torch
import torch.nn as nn
from configs.config_loader import YOLOConfig, load_config
from data.utils.df_utils import create_df
import torchvision.models as models
from torchvision.models import VGG16_Weights

"""
    architecture_config contains the layers of the YOLO v1 architecture, it doesn't include the fully connected layers (fc).

    Tuple: (kernel_size, num_filters, stride, padding) -- Convolution Layer.
        - Convolution layers extract feature maps from the image data.

    "M" is maxpooling with stride and kernel size of 2.
        - Pooling layers down-sample the tensors. i.e make the tensors smaller as it passes thru the network but keep the feature maps.
"""
architecture_config = [
    # --- Conv Layer ↓: 7 is the kernel size, 64 is the number of filters, 2 is the stride, 3 is the padding.
    (7, 64, 2, 3),
    "M",  # maxpool
    (3, 192, 1, 1),  # conv layer...
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        4,  # --- Repeat the above (2-lines/conv layers) 4 times.
    ],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        2,  # --- Repeat the above (2-lines/conv layers) 2 times.
    ],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


def calculate_class_confidence(
    class_probs: torch.Tensor, confidence_scores: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the final class-specific confidence score for each bounding box by multiplying the conditional class probability and the predicted confidence score.

    This is based on the formula from the YOLOv1 paper:
        Pr(Class_i | object) * Pr(Object) * IoU = Pr(Class_i) * IoU

        It is described in the paper at 2. Unified Detection section:
            "At test time we multiply the conditional class probabilities and the individual box confidence
            predictions [Formula] which gives us class-specific confidence scores for each box. These scores encode
            both the  probability of that class appearing in the box and how well the predicted box fits the object."

        - Pr(Class_i | object): The model's classification output for a specific class i, given that an object is present in the grid cell.
        - Pr(Object): The model's prediction of whether an object exists in a grid cell. This is the pc (confidence score) you get from the bounding box prediction head.
        - IoU: The Intersection over Union of the predicted bounding box and the ground-truth box. The paper notes that at test time, the model's confidence prediction C is supposed to be equal to this IoU.

    Args:
        class_probs (torch.Tensor): The conditional class probabilities,
                                    shape (N, S, S).
        confidence_scores (torch.Tensor): The individual box confidence scores (pc),
                                          shape (N, S, S, B).

    Returns:
        torch.Tensor: The final class-specific confidence scores,
                      shape (N, S, S, B).
    """

    # Unsqueeze class_probs to match the dimensionality of confidence_scores
    expanded_class_probs = class_probs.unsqueeze(-1)  # shape (N, S, S, 1)

    # Multiply to get the final score for each box
    final_scores = confidence_scores * expanded_class_probs

    return final_scores


def apply_yolo_activation(pred: torch.Tensor, C: int, B: int) -> torch.Tensor:
    """
    Applies the necessary activation functions to the raw YOLO predictions (pc, x, y), used when post-processing but not for computing loss.

    Confidence Scores: A probability must be in the range [0,1]. Using a sigmoid function on the raw output forces the model's prediction into this valid probability range. The loss function can then train the model to output a value that, after passing through sigmoid, is close to 1 for an object and 0 for no object.
    Bounding Box Center Coordinates (x,y): The YOLOv1 paper parameterize the x and y coordinates of the bounding box to be an offset relative to the top-left corner of the grid cell. This means that x and y must also be within the range [0,1]. Similar to the confidence score, applying a sigmoid function to the raw x and y predictions naturally constrains them to this range, which is critical for correctly calculating the bounding box's position relative to the grid cell. Without this constraint, the model's predictions could be outside the grid cell, leading to incorrect localization.

    Args:
        pred (torch.Tensor): The raw model output tensor.
        C (int): The number of classes.
        B (int): The number of bounding boxes per grid cell.

    Returns:
        torch.Tensor: The activated prediction tensor.
    """
    # Create a copy to avoid modifying the original tensor.
    pred_activated = pred.clone()

    # Apply sigmoid to class scores to get probabilities in the [0, 1] range.
    pred_activated[..., :C] = torch.sigmoid(pred_activated[..., :C])

    # Apply sigmoid to confidence and x, y coordinates for each of the B bounding boxes. Works well for any B value.
    for b in range(B):  # Dynamically for each box in a cell.
        start_idx = C + b * 5
        # Index 0 is confidence (pc), and indices 1-2 are x, y coordinates.
        # Note: w, h are left as-is.
        pred_activated[..., start_idx : start_idx + 3] = torch.sigmoid(
            pred_activated[..., start_idx : start_idx + 3]
        )
        # # Bounding bbox_1: confidence (pc) and x, y coordinates
        # pred_activated[..., C : C + 2] = torch.sigmoid(pred_activated[..., C : C + 2])

        # # Bounding bbox_2: confidence (pc) and x, y coordinates
        # pred_activated[..., C + 5 : C + 7] = torch.sigmoid(
        #     pred_activated[..., C + 5 : C + 7]
        # )

    return pred_activated


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        **kwargs
    ):
        """
        Convolution Layer Block

        Args:
            in_channels (int) : Number of channels from the input ex: image -> 448x448x3 -> 3 is the in_channel.
            out_channels (int): Number of channels that wil be outputted.
            kernel_size (int): The size of the filter matrix that will be computed with the input, the filter matrix does a sling-window over the input matrix, the result is returned this layer's output matrix. This creates feature maps.
            stride (int): When the kernel matrix slides over the input, after each time it computes with the input, it will skip a stride of pixels to compute the next pixel area
            padding (int): Padding pixels is applied to the input matrix on all four sides, the padding and stride will determine the size of the output.
        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,  # bias set to False because we are using batchnorm.
            **kwargs
        )
        # NOTE: batch normalization wasn't used in the original paper because it wasn't invented yet. It normalize the feature maps across the batch dimension to stabilize and speed up training.
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leakyReLU = nn.LeakyReLU(0.1)  # Activation function.

    def forward(self, input):  # Call function.
        return self.leakyReLU(self.batch_norm(self.conv(input)))


class YOLOv1(nn.Module):
    def __init__(
        self,
        cfg: YOLOConfig,
        in_channels: int = 3,
        use_pre_trained_backbone: bool = False,
        *args,
        **kwargs
    ):
        """
        YOLO v1 model.

        NOTE: I included the VGG pre-trained model when use_pre_trained_backbone=True, else the model will be identical to the YOLOv1 paper.

        Args:
            cfg (YOLOconfig): Project configurations.
            in_channels (int) : Number of channels from the input image -> 448x448x3, 3 is the in_channels.
            use_pre_trained_backbone (bool): Whether to use the CNN layers of the pretrained VGG16 model.
        """

        super(YOLOv1, self).__init__()
        self.cfg = cfg
        self.use_pre_trained_backbone = use_pre_trained_backbone

        if self.use_pre_trained_backbone:
            # Load the VGG16 model with its pre-trained ImageNet weights
            self.vgg16_backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.backbone = self.vgg16_backbone.features

            # Freeze the backbone and only train your new fully connect layers layers.
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.in_channels = in_channels
            self.conv = self._create_conv_layers()

        # Create the fully connected layers
        self.fc = self._create_fc(**kwargs)

    def forward(self, input):  # Call function.
        # --- 1: Pass the image through the convolution layers.
        if self.use_pre_trained_backbone:
            out = self.backbone(input)
        else:
            out = self.conv(input)

        # --- 2: Pass the last convolution layer's output through the fully connected layers.
        return self.fc(torch.flatten(out, start_dim=1))

    def _create_conv_layers(self):
        """Create the convolution layers."""
        layers = []

        in_c = self.in_channels

        for x in architecture_config:
            # --- Handle one conv layer i.e -> (7, 64, 2, 3)
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels=in_c,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
                # Update in_c to be on the pervious tuple's num_filters. On the first run its on the images channel size (3), second run it will be on 64 -> third: 192, etc..
                in_c = x[1]
            # --- Handle pooling layers "M".
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # --- Handle list layers.
            elif type(x) == list:
                conv_1 = x[0]
                conv_2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels=in_c,
                            out_channels=conv_1[1],
                            kernel_size=conv_1[0],
                            stride=conv_1[2],
                            padding=conv_1[3],
                        ),
                        CNNBlock(
                            in_channels=conv_1[1],
                            out_channels=conv_2[1],
                            kernel_size=conv_2[0],
                            stride=conv_2[2],
                            padding=conv_2[3],
                        ),
                    ]
                    # Update in_c to be conv_2 num_filters.
                    in_c = conv_2[1]
        return nn.Sequential(*layers)

    def _create_fc(self, **kwargs):
        """Create the fully connected layers (FC)."""
        cfg = self.cfg
        S, B, C = cfg.S, cfg.B, cfg.C

        if self.use_pre_trained_backbone:
            # NOTE: The output of VGG16's features for a 448x448 image is (512, 14, 14)
            return nn.Sequential(
                nn.Flatten(),
                #  VGG16 backbone for a 448×448 input is a tensor of shape 512×14×14. Therefore, the first FCN layer must accept an input of (512 * 14 * 14)
                nn.Linear(512 * 14 * 14, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, S * S * (B * 5 + C)),
            )
        else:
            # Use the yolov1 paper fully connected layers (1024, 7, 7).
            return nn.Sequential(
                nn.Flatten(),  # flatten the output from the last convolution layer.
                nn.Linear(1024 * S * S, 4096),  # --- First FC layer
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(  # --- Second FC layer
                    4096,
                    S * S * (B * 5 + C),
                    # From paper last output shape: S * S * ( B * 5 + C)
                ),
                # The paper applies a activation 'like' function to the output @ confidence = pc * best_prob -> implemented in extract_and_convert_pred_bboxes()
            )


# Test as module:
#        python -m model.yolov1
def test():
    cfg = load_config("config_voc_dataset.yaml")
    yolo = YOLOv1(cfg=cfg, in_channels=3)

    # Test with 2 images examples
    x = torch.randn((2, 3, 448, 448))
    print(yolo(x).shape)

    # Output shape: (2, 1470), 1470-> LABEL_NODES


if __name__ == "__main__":
    test()
