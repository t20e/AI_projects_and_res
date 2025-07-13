
# root .py file 

import torch
import torchvision.transforms as T
from argparse import Namespace
from yolov1.model import YOLOv1
from yolov1.train import train
import torch.optim as optim
from utils.checkpoints import load_checkpoint

torch.manual_seed(21) # seed

# Dataset structure for one cell -> [with_mask, without_mask, mask_worn_incorrectly, pc_1, x, y, w, h, pc_2, x, y, w, h]. pc_1 is probability  score


# <------------- Hyperparameters/Config ------------->
config = Namespace(
    DEVICE = torch.device("mps"), # apple silicon M series
    NUM_WORKERS = 2,
    PIN_MEMORY = True,
    
    EPOCHS = 50,
    LEARNING_RATE = 2e-5,
    BATCH_SIZE = 16, #64,
    WEIGHT_DECAY = 0, #TODO plasy with wight decay

    # load a model with weights that u have been trained to train it more
    LOAD_MODEL = False,
    LOAD_MODEL_FILE = "./checkpoints/Yolov1_facemask_objectDetection_epoch50_2025-04-09-18h_31m.pt",
    
    DATASET_DIR = "./data", # root path to the dataset dir
    IMAGE_SIZE = 448,

    C = 3, # how many classes in the dataset
    B = 2, # how many bounding boxes does the model perdict per cell
    S = 7, # split_size, how to split the image, 7x7=49 grid cells,
    IOU_THRESHOLD = 0.5, # the iou threshold when comparing bounding boxes for NMS
    MIN_THRESHOLD = 0.4, # the minimal confidence to keep a predicted bounding box
    PLOT_WHILE_TRAINING = True # bool to plot true and predicted bounding boxes on the images while training, this plot happens at random epochs
)

config.NUM_NODES_PER_CELL = config.C + 5 * config.B # The total number of nodes per cell, which would be the size ==> [with_mask, without_mask, mask_worn_incorrectly, pc_1, bbox1_x_y_w_h, pc_2, bbox2_x_y_w_h] = 13 nodes.
config.NUM_NODES_PER_IMG = config.S * config.S * (config.C + config.B * 5) # number of nodes that each image has. If S=7 C=3 B= 2 ==+> 7*7 * (3 + 2 * 5) = 637, also 13 * 49 = 637


# <------------- Transforms ------------->
class Compose(object):
    """Resize, normalize, and transform to tensor."""
    def __init__(self, transforms):
        self.T = transforms
    def __call__(self, img, bboxes):
        for t in self.T:
            img, bboxes = t(img), bboxes
        return img, bboxes
    
transforms = Compose([
    T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    T.ToTensor()
])



def main(config):
    """Root function"""
    model = YOLOv1(in_channels=3, S=config.S, B=config.B, C=config.C).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    if config.LOAD_MODEL:
        # print("\n\nLoading Model.")
        load_checkpoint(config.LOAD_MODEL_FILE, model, optimizer)
        
    
    train(config=config, model=model, optimizer=optimizer, transforms=transforms)

if __name__ == "__main__":
    print("Running Root.")
    main(config)
