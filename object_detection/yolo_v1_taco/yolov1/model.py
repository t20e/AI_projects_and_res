"""YOLO v1 model"""

import torch
import torch.nn as nn


"""
    Architecture config contains the layers of the YOLO v1 architecture, it doesn't include the fully connected layers (fc).

    Tuple: (kernel_size, num_filters, stride, padding)

    "M" is maxpooling with stride and kernel size 2

    List: [
            (kernel_size, num_filters, stride, padding),
            (kernel_size, num_filters, stride, padding),
            num_times_to_repeat
        ]
"""
architecture_config = [
    # Conv Layer, 7 is the kernel size, 64 is the number of filters, 2 is the stride, 3 is the padding
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
        4,  # repeat that the above (2 lines/conv layers) 4 times.
    ],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
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
        Convolutional Layer Block

        Parameters
        ----------
            in_channels : (int)
                Number of channels from the input, ex: image -> 448x448x3 3 is the in_channels
            out_channels : (int)
                Number of channels that will be outputted
            kernel_size : (int)
                States the size of the filter matrix that will be computed with the input, the filter matrix does a sling-window over the input matrix, the result is returned in the output matrix. This recognizes patterns from an image/data. This is the convolutional computation.
            stride : (int)
                If stride is 2, the output will be halved. When the kernel/filter matrix slides over the input, after each time it computes with the input, it will skip a stride of pixels to compute the next convolution.
            padding : (int)
                Padding is applied to the input matrix on all four sizes, this preserve size of the ouput.

        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            **kwargs
        )  # bias set to False because were using batchnorm
        self.batch_norm = nn.BatchNorm2d(
            out_channels
        )  # batchnorm wasn't used in the original paper because it wasn't invented yet, It normalizes the feature maps across the batch dimension to stabilize and speed up training.
        self.LeakyReLU = nn.LeakyReLU(0.1)  # Activation function

    def forward(self, x):  # Call function
        return self.LeakyReLU(self.batch_norm(self.conv(x)))


class YOLOv1(nn.Module):

    def __init__(self, in_channels: int, S: int, B: int, C: int, *args, **kwargs):
        """
        Yolo v1 Model

        Parameters
        ----------
            in_channels : (int)
                Number of channels from the input, ex: image -> 448x448x3, 3 is the in_channels
            S : int
                The split size of the row/col grid. 7 => 7x7= 49 grid cell.
            B : (int)
                How many bounding boxes does each cell predict.
            C : (int)
                Number of classes from the dataset.
        """
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.S = S
        self.B = B
        self.C = C
        self.conv = self._create_conv_layers()
        self.fc = self._create_fc(**kwargs)

    def forward(self, input):  # call method

        # pass thru all the Convolutional layers
        out = self.conv(input)
        # # pass thru fully connected layers/dense neural network
        out = self.fc(torch.flatten(out, 1))

        out = out.view(input.size(0), self.S, self.S, 28)
        C = self.C
        act = torch.sigmoid
        out[..., C] = act(out[..., C])  # pc1
        out[..., C + 1 : C + 3] = act(out[..., C + 1 : C + 3])  # x1,y1
        out[..., C + 3 : C + 5] = act(out[..., C + 3 : C + 5])  # w1,h1
        out[..., C + 5] = act(out[..., C + 5])  # pc2
        out[..., C + 6 : C + 8] = act(out[..., C + 6 : C + 8])  # x2,y2
        out[..., C + 8 : C + 10] = act(out[..., C + 8 : C + 10])  # w2,h2
        return out

    def _create_conv_layers(self):
        """Create the convolutional layers"""
        layers = []
        in_c = self.in_channels

        for x in architecture_config:
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
                # when we fist run it we start at 3 which is the images channel size, then 64, 192, etc..thru architecture, this line just updates it.
                in_c = x[1]
            elif type(x) == str:  # "M" maxpool
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:  # repeats layers
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_c,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        ),
                        # ]
                        # layers +=[
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        ),
                    ]
                    # update in_channels to be conv2 from conv1 to be the input of the next layer
                    in_c = conv2[1]
        return nn.Sequential(
            *layers  # *layers is to unpack the list of layers and convert it into a sequential.
        )

    def _create_fc(self, **kwargs):
        """
        Create the fully connected layers/dense neural network
        """
        S, B, C = self.S, self.B, self.C

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(
                # From paper: S * S * ( B * 5 + C)
                4096,
                S * S * (B * 5 + C),
            ),
        )
