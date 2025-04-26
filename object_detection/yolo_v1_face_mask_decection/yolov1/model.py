# Implement YOLOv1 model architecture

"""
YOLOv1 Model
"""


import torch
import torch.nn as nn


"""
    Architecture config contains the layers of the YOLO v1 architecture, it doesn't include the fully connected layers fcs.

    Tuple: (kernel_size, num_filters, stride, padding)
    "M" is maxpooling with stride and kernel size 2
    List: [(kernel_size, num_filters, stride, padding), (kernel_size, num_filters, stride, padding), num_times_to_repeat]
"""
architecture_config = [
    (7, 64, 2, 3), # Conv Layer 7 is the kernel size, 64 is the number of filters, 2 is the stride, 3 is the padding
    "M",  
    (3, 192, 1, 1), # conv layer...
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # the last 4 means repeat that layer in the () 4 times
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

class CNNBlock(nn.Module):
    # c = CNNBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs) # bias set to False because were using batchnorm
        self.batch_norm = nn.BatchNorm2d(out_channels)# batchnorm wansnt used in the original paper becuase it wasnt invented yet, It normalizes the feature maps across the batch dimension to stabilize and speed up training.
        self.leakyreul = nn.LeakyReLU(0.1) # Activation function
        
    def forward(self, x): # Call function
        return self.leakyreul(self.batch_norm(self.conv(x)))


class YOLOv1(nn.Module):
    """
    Yolov1 Model
    
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
    def __init__(self, in_channels, S, B, C, *args, **kwargs):
        super(YOLOv1, self).__init__()
        self.in_channels= in_channels
        self.S = S
        self.B = B
        self.C = C
        self.conv = self._create_conv_layers()
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, input):# call method
        x = self.conv(input) # pass thru all the Convolutional layers
        return self.fcs(torch.flatten(x, start_dim=1)) # pass thru fully connected layers/dense neural network
    
    def _create_conv_layers(self):
        """Create the convolutional layers"""
        layers = []
        in_c = self.in_channels
        
        for x in architecture_config:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels=in_c, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                ]
                in_c = x[1] # when we fist run it we start at 3 which is the images channel size, then 64, 192, etc..
            elif type(x) == str: # "M" maxpool
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list: # repeats layers
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(in_c, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]),
                    # ]
                    # layers +=[
                        CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])
                    ]
                    in_c = conv2[1] # update in_channels to be conv2 from conv1 to be the input of the next layer
        return nn.Sequential(*layers) # *layers is to unpack the list of layers and convert it into a sequential.
    
    
    def _create_fcs(self, **kwargs):
        """
        Create the fully connected layers/dense neural network
        """
        S, B, C = self.S, self.B, self.C
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)) # NOTE: We will use the same function for the true labeled tensor and predicted model ouput tensor, so we will make them the same shape, thats why we add + 5 @ self.C + 5 * self.B below
        )
        
        
# def test(S = 7, B=2, C=3):
#     model = YOLOv1(in_channels=3, S=S, B=B, C=C)
#     x = torch.randn((2, 3, 448, 448)) # 2 image examples 
#     print(model(x).shape)
#     # Example: 7*7*13=637 -> output should be torch.Size([2, 637])
    
# test()