# Liraries
import torch
import os
import pandas as pd
from PIL import Image
from typing import Optional
from torchvision.transforms import Compose
from typing import Tuple
import sys

# My modules
from data.utils.setup_transforms import setup_transforms
from data.utils.df_utils import create_df
from configs.config_loader import YOLOConfig, load_config
from data.utils.extract_annotations import extract_annotations


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: YOLOConfig, transforms: Optional[Compose] = None):
        """
        Dataset Class: Structures the images and annotations.

        Note: Dataset format must be in corner-points format, which will then be converted in __getitem__().

        Args:
            cfg (YOLOConfig): Model configurations.
            transfroms (torchvision.transforms): Transformations to apply to images and its annotations/labeled data.
        """
        self.cfg = cfg
        self.transforms = transforms

        self.dataset_dir = os.path.join(os.getcwd(), "datasets", cfg.DATASET)
        # Load a dataframe
        self.annotations = create_df(
            dataset_path=self.dataset_dir, num_to_load=cfg.NUM_IMAGES
        )

    def __len__(self) -> int:  # Returns the size of the dataset
        return len(self.annotations)

    def __getitem__(self, index) -> Tuple[Image.Image, torch.Tensor]:
        """
        Get a single image and its corresponding label matrix.

        Args:
            index (int): The index of the image in the csv dataframe list.
        Returns:
            tuple ( PIL.image, tensor) :
                - (Single image, label bounding box matrix).
                - The label tensor is of shape (S, S, CELL_NODES).
        """
        cfg = self.cfg
        S, B, C = cfg.S, cfg.B, cfg.C

        # --- 1 Make sure that whatever the index is its not greater than the size of the dataframe
        if index >= cfg.NUM_IMAGES:
            print(f"\n\nERROR: Index is out of bounds. Index: {index}, size of dataframe: {self.__len__()}, index starts at zero. \n\nOccurred at: VOCDataset.__getitem__()")
            sys.exit(0) # exit program


        # --- 2: Get the .xml file path from that dataframe with the index.
        label_xml_path = os.path.join(self.dataset_dir, "Annotations", self.annotations.iloc[index, 1])

        # --- 3: Get the corner-points bboxes from the annotations.
        bboxes_corner_points = extract_annotations(cfg=cfg, xml_path=label_xml_path)

        # --- 4: Get the image
        img_path = os.path.join(self.dataset_dir, "JPEGImages", self.annotations.iloc[index, 0])
        img = Image.open(img_path)
        print(img)

        # --- 5: Convert coordinates from corner-points to mid-point


def test():
    cfg = load_config("yolov1.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)
    d = VOCDataset(cfg, t)
    d.__getitem__(0)


test()
