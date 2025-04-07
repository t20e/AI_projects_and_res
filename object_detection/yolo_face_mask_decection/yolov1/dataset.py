# Dataset class

from utils.plot import plot_bbox_on_img, visualize_grid_on_img
from utils.convert_coordinates import voc_to_yolo, yolo_to_voc
from utils.parse_xml import parse_voc_annotation


import torch
import os
from PIL import Image
import pandas as pd

import gc # python built-in memory clean-up
# !pip install psutil
# import psutil

# CLASSES = ["with_mask", "without_mask", "mask_worn_incorrectly"]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, S, B, C, transform):
        """
        Dataset class 
        
        Parameters
        ----------
            dataset_dir (str): path to dataset directory that contains ./images and ./labels
            S (int): split_size, how many cell to split image into, ex: S=7 -> 7x7=49 cells
            B (int): how many boxes does each cell predict
            C (int): how many classes are we predicting, [with_mask, without_mask, mask_worn_incorrectly]
            transform (torchvision.transforms): resize and normailize images 
        Variables
        ---------
            self.annotations (pandas.dataframe): contains a dataframe with corrisponding names for the labels and images
            self.img_dir (str): path to image directory
            self.label_dir (str): path to label directory
            
        Note
        ----
            Classes-> ["with_mask" = idx_0, "without_mask" = idx_1, "mask_worn_incorrectly" = idx_2].s
            
            The mask dataset bounding boxes are in PASCAL VOC format so we convert it to YOLO format. PASCAL VOC format from absolute coordinates "corner-points" [Xmin, Ymin, Xmax, Ymax] to YOLO format "midpoint" [x_center, y_center, height, width].
    
        """
        self.create_annotations_csv_file()
        self.annotations = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
        self.img_dir = os.path.join(dataset_dir, "images")
        self.label_dir = os.path.join(dataset_dir, "labels")
        self.S = S 
        self.B = B 
        self.C = C
        self.transform = transform 
        
    def __len__(self): # returns size of the entire dataset
        return len(self.annotations)
    
    def __getitem__(self, index, plot_img=None):
        """
        Grab a single image and its corresponding label matrix.
        
            Parameter
            ---------
                index: (int)
                    the index of the image in the csv dataframe list
                num_img_to_plot: (int)
                    number of images to plot the bbox and cell that the bbox belongs to, just to debug we can see the images and bbox as we load the dataset
            Returns
            -------
                tuple (PIL.Image, pytorch.tensor): single image, label matrix bbox for that image
        """
        # grab the label .xml file
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # Open the image
        image = Image.open(os.path.join(self.img_dir, self.annotations.iloc[index, 0])).convert("RGB")
        
        voc_label = parse_voc_annotation(label_path, num_classes=self.C)
        
        bboxes = []
        for voc in voc_label:
            bboxes.append(voc_to_yolo(image.size, voc))
        # convert bboxes from list to tensor
        bboxes = torch.tensor(bboxes)

        if self.transform: # apply transform
            # NOTE you can resize the image with no problems because the bounding boxes are percentages of the image and will resize correctly. But only if its in YOLO format.
            image, bboxes = self.transform(image, bboxes) # performs resize, and normalization

        if plot_img:
            plot_bbox_on_img(image, bboxes)
            
        # NOTE: We will use the same function for the true labeled tensor and predicted model ouput tensor, so we will make them the same shape, thats why we add + 5 @ self.C + 5 * self.B below
        # Shape: example S=7, C=3, B=2 = (7, 7, 13) -> [with_mask, without_mask, mask_worn_incorrectly, pc_1, bbox1_x_y_w_h, pc_2, bbox2_x_y_w_h] = 13 nodes -> each cell will have 13 nodes.
        # We just wont use the pc_2/probability_score and bbox2_x_y_w_h in the true labeled_matrix, but the model output will predict two bboxes per cell.
        label_matrix = torch.zeros(self.S, self.S, self.C + 5 * self.B) # if S=7, C=3,B=2 torch.Size([7, 7, 13])
        
        bbox_midpoint_cells = []
        # Make the bboxes relative to the cell its mid-point is in, instead of the image and add it to the label_matrix
        for box in bboxes:
            c1, c2, c3, x, y, width, height = box.tolist()

            # find which cell the bbox's mid-point is in
            # i, j represents (row, col), the cell location that contains the bbox's mid-point
            # NOTE: If you want to see a plot of it use visualize_grid_on_img(image, cell=(i, j), split_size=7)
            i, j = int(self.S * y), int(self.S * x)
            if plot_img:
                bbox_midpoint_cells.append((i, j))
                
            #NOTE: Now we make the bbox coordinates relative to the cell its in, instead of being relative to the entire image. Note x, y cant not be bigger than 1 that would mean its larger than the cell, however the height and width of the bbox can be bigger than 1.
            x_rel_cell, y_rel_cell = self.S * x - j, self.S * y - i
            width_rel_cell, height_rel_cell = width * self.S, height * self.S
            
            if label_matrix[i, j, self.C] == 0: # checking if theres currently no object in i and j, this is also the position of the first probabiltiy_score
                # NOTE: if the split_size is low like 7, an image where two people are close to each, and one is wearing a mask and the other is not, the label matrix will only take one of them becuase they will likely occupy the same cell, and each cell label data has only one bounding box
                label_matrix[i, j, self.C] = 1 # this cell is now taken
                box_coordinates = torch.tensor(
                    [x_rel_cell, y_rel_cell, width_rel_cell, height_rel_cell]
                )

                # Add the bbox coordinates at the bbox1_x_y_w_h in the label matrix
                label_matrix[i, j, self.C + 1 : self.C + 1 + 4] = box_coordinates
                
                # Add the classes label to the first 3 nodes
                label_matrix[i, j, :3] = torch.tensor([c1, c2, c3])
        
        if plot_img:
            visualize_grid_on_img(image, cells=bbox_midpoint_cells, split_size=self.S)

        return image, label_matrix
        
        
    def create_annotations_csv_file(self):
        # Creates a dataframe with matched file_names for labels and corresponding images and saves it to a csv file
        if os.path.exists("./data/train.csv"): # check if we have already created the file
            print("CSV file exists!")
            return
        
        
        imgs_dir = "./data/images"
        label_dir = "./data/labels"

        # Get all image and label filenames
        image_files = sorted([f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))])
        label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
        # Match files by name
        matched_data = []

        for image_file in image_files:
            label_file = image_file.rsplit('.', 1)[0] + '.xml'  # assuming labels are .txt
            if label_file in label_files:
                matched_data.append({
                    "train":  image_file,
                    "label": label_file
                })
        df = pd.DataFrame(matched_data)
        df.to_csv("./data/train.csv", index=False)
        del df # deletes dataframe to save memory
        gc.collect() # tells python to clean up-memory
        