'''
    This file contains the main model for the YOLO V1 Model



'''


import torch
import torch.nn as nn


'''
    Architecture config contains the layers of the YOLO V1 Archeticure, see in the readme.md at line-item 6. It doesnt include the fully connected layers.


    Tuple: (kernel_size, num_filters, stride, padding)
    "M" is maxpooling with stride and kernel size 2
    List: [(kernel_size, num_filters, stride, padding), (kernel_size, num_filters, stride, padding), num_times_to_repeat]
'''
architecture_config = [
    
    
    (7, 64, 2, 3), # Conv Layer 7 is the kernel size, 64 is the number of filters, 2 is the stride, 3 is the padding
    "M", 
    (3, 192, 1, 1), # conv layer
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
    '''
        This is a CNN block that we will use many times 
        
    '''
    
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # bias is set to false because we are using batchnorm
        self.batchnorm = nn.BatchNorm2d(out_channels) # also batchnorm wansnt used in the original paper becuase it wasnt invented yet
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(nn.Module):
    
    
    '''
    
    Yolov1 model
    
    Note: model has an output shape of (batch_size=1, grid_size=7, grid_size=7, channels=30), the channels 30 are for the 20 class predictions + 2 * ( 5 for (probabilty that theres an object in the cell, and the other 4 for the bounding boxes predictions))

    '''
    
    def __init__(self, in_channels=3, **kwargs): # in_channels is the number of channels in the input image so 3 for RGB
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs) # fcs is fully connected layer
    
    def forward(self, x):
        x = self.darknet(x)
        # flatten then send to fully connected layer
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                           ]
                in_channels= x[1]
                
            elif type(x) == str: # meaning = "M"
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0] # tuple
                conv2 = x[1] # tuple
                num_repeats = x[2]  # int
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]
                            )
                    ]
                    
                    layers += [
                        CNNBlock(
                            conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1] # now we change in_channels from conv1 to conv2 so that it will be the input for the next layer when we run this loop
                    
        return nn.Sequential(*layers) # *layers is to unpack the list of layers and convert it to a nn.Sequential
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        ''' 
            split_size = (SxS), meaning if we have S = 7 we have a 7x7 = 49 grid
        '''
        S, B, C = split_size, num_boxes, num_classes
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096), # NOTE: the orginal paper it was 4,096 instead of 496 but that would take a long time to train
            nn.Dropout(0.0), # dropout was 0.5 in the original paper
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)) # the orginal paper it was 4,096 instead of 496 but that would take a long time to train, this will be reshaped to be ( S, S, 30) C+B*5 = 30
        )
        
def test(S = 7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448)) # 2 image examples 
    print(model(x).shape)
    # output should be torch.Size([2, 1470])
    # 7*7*30=1470
    
# test()