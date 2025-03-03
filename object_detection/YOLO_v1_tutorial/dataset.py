import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        
        boxes = []
        
        # open the text file
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                
                boxes.append([class_label, x, y, width, height])
        
        # open the images 
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)
            
        # just  note here that the additional 5 nodes  * self.B is not going to be used, the only thing thats going to be used is the 25 first nodes, however another function assumes that this will be 30 so we add * self.B
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B ))
        
        # we need to convert everything to fit the label_matrix
        for box in boxes:
            # we need to see which cell the bounding box belongs to and then we need to convert the bounding box to be relative to that cell 
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            
            # x_cell relative to the cell, y_cell relative to the cell
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            width_cell, height_cell = (
                # width and height of the bounding box relative to the entire image, then we scale it by self.S to be relative to the cell, self.S is the number of cells in the image both horizontally and vertically, num of grids.
                width * self.S,
                height * self.S,
            )
            
            if label_matrix[i, j, 20] == 0: # checking if theres currently no object in i and j
                label_matrix[i, j, 20] = 1 # this cell is now taken
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1 # specify the class label, so which object it is
                
        return image, label_matrix