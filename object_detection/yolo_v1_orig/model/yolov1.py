"""YOLO v1 Model Architecture"""

import torch
import torch.nn as nn
from configs.config_loader import YOLOConfig, load_config
from data.utils.df_utils import create_df

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
    def __init__(self, cfg: YOLOConfig, in_channels: int = 3, *args, **kwargs):
        """
        YOLO v1 model.

        Args:
            cfg (YOLOconfig): Project configurations.
            in_channels (int) : Number of channels from the input image -> 448x448x3, 3 is the in_channels.
        """
        super(YOLOv1, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.conv = self._create_conv_layers()
        self.fc = self._create_fc(**kwargs)

    def forward(self, input):  # Call function.
        # --- 1: Pass the image through the convolution layers.
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
