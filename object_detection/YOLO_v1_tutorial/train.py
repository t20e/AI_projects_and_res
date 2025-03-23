
'''
    How to improve: pre-train on resNet etc..
    
    To run this model: python train.py
'''

import torch 
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
import time
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    save_model,
    load_model #my loader below
)
from loss import YoloLoss

import random


"""
    # NOTE: I made a lot of notes on these util functons in live_feed_yolo_model.ipynb
    and also in the res repo in .coding.res/AI.res/object_detection/YOLO.res has more resources and notes
"""

# seed = 123 # 123 seed to follow tutorial
seed = random.randint(1, 200)
torch.manual_seed(seed) # this is so we get the same seed as the turorial

# Hyperparameters etc.
LEARNING_RATE = 2e-5


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps" # to use the M1 MAC GPU
DEVICE = torch.device('mps')# to use the M1 MAC GPU

# NOTE: if u use a BATCH_SIZE > NUM_OF_IMAGES_TO_TRAIN/TEST u will get a divisible by zero error
BATCH_SIZE = 16 # 64 in the original paper but that takes a lot of vram, reducing to 16
WEIGHT_DECAY = 0 # 0 because it would take a long time to train, the original paper had pre-trained on ImageNet for 2 weeks, then they trained on out dataset for a long time, they also did it with very heavy data augmentation
# To overfit more easily we will not use regularization
# EPOCHS = 25 #100 
EPOCHS = 150  #100  50  25 
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True # load in a trained model that was saved with checkpoint
# NOTE: some of the models were train on 496 inputs vs some at 4096, the model class needs to be changed to the same as the model that is used this is for the FULLY CONNECTED LAYERS in the model at _create_fcs()
LOAD_MODEL_FILE = "./saved_models/overfitted-YoloV1-train-on-100img-and-4096.pt" 
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"



class Compose(object):
    """Resize and transform to tensor"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes): # bboxes = bounding boxes
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

# one thing that can improve this is that u do a normalization. mean = 0 and standard deviation = 1
transforms = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor() # convert to tensors and NOTE: normalizes it under the hood
])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        # update the progress bar
        loop.set_postfix(loss=loss.item())
    
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()
    
    if LOAD_MODEL: # if we have a model saved we can load it
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        
        # my code below to load a model above is orginal
        # model = load_model(
        #     "YoloV1-train-on-entire-dataset-and-4096",
        #     # "YoloV1",
        #     # "YoloV1-TEST",
        #     Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE),
        #     DEVICE
        # )
    
    train_dataset = VOCDataset(
        # first overfit on the 8examples.csv then train on the 100examples.csv
        # "data/8examples.csv",
        "data/100examples.csv",
        # "./data/train.csv",
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    
    test_dataset = VOCDataset(
        # "data/train.csv"
        # first overfit on the 8examples.csv then train on the 100examples.csv
        "data/test.csv",
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True, # this is when we have a batch that has only 2 examples, we can drop that batch so we dont do a gradient update on a batch that is not full!, in the example of using 8example.vsc and we have a batch size of 16, then it will always drop it because 8 is not divisible by 16 this will give an error so when using 8examples.csv we need to set drop_last to False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    
    for epoch in range(EPOCHS):
        print("\n\n" + "|" + "-" * 64 + "|")
        print("Epoch:", epoch + 1)
        
        
        for x, y in train_loader:
            '''
            This for loop will print the predicted bounding boxes on the images as the model trains
            in the example he trained the model until it got a mAP of 0.9 then he used that model to predict the bounding boxes on the images with this for loop.
            '''
            x = x.to(DEVICE)
            # x.shape = torch.Size([16, 3, 448, 448]) # 16 is num of batches

            ##plot the images
            for idx in range(BATCH_SIZE): 
                print(x.shape)
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                # print("\n\n")
                # for box in bboxes: #test
                #     print(box[2:])
                plot_image(x[idx].permute(1, 2, 0).to(DEVICE), bboxes)
            sys.exit()
        
        
        
        
        # grab the bounding boxes from the predicted output and the true labels
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )
        
        # compute mean average percision between the predicted and true bounding boxes
        mean_average_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        
        print(f"\nTrain mAP: {mean_average_prec}")
        
        # checkpoint save the model if mean average precision is greater than 0.9
        # if mean_average_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)
        
        
        train_fn(train_loader, model, optimizer, loss_fn)
    
    # save the model
    # save_model(model, "YoloV1")
    save_model(model, "a1-4096-100-images-to0-0.90-mAP")
    # save_model(model, "YoloV1-train-on-entire-dataset-and-4096")
    # save_model(model, "YoloV1-on-entire-dataset")
        
        
if __name__ == "__main__":
    main()
        