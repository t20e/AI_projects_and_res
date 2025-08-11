# Liraries
import torch
import os
import pandas as pd
from PIL import Image
from typing import Optional

from typing import Tuple
import sys

# My modules
from configs.config_loader import YOLOConfig, load_config
from data.utils.setup_transforms import setup_transforms
from data.utils.df_utils import create_df

from data.utils.setup_transforms import CustomCompose
from data.utils.VOC_xml_extraction_pipeline import VOCAnnotationsExtraction


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: YOLOConfig,
        which_dataset: str,
        num_samples: int = 10,
        transforms: Optional[CustomCompose] = None,
        print_df: bool = False,
    ):
        """
        Dataset Class: Structures the images and annotations.

        Note: The dataset format must be in corner-points format, which will then be converted in __getitem__() to mid-points with normalized percentage values.

        Args:
            cfg (YOLOConfig): Model configurations.
            which_dataset (str): Name of dataset to grab "VOC2012_train", "VOC2012_val" or "VOC2012_test"
            num_samples (int): The number of samples (images/labels) to load from the dataset and create a dataframe.
            transforms (torchvision.transforms): Transformations to apply to images and its annotations/labeled data.
        """
        self.cfg = cfg
        self.transforms = transforms
        self.num_samples = num_samples
        self.dataset_path = os.path.join(os.getcwd(), "data/datasets", which_dataset)
        # Load a dataframe
        self.annotations = create_df(
            cfg=cfg,
            dataset_path=self.dataset_path,
            num_to_load=num_samples,
            save_to_csv=False,
        )
        if print_df:
            print("\nDataframe:", self.annotations, "\n")

        self.voc_extraction = VOCAnnotationsExtraction(cfg=cfg, transforms=transforms)

    def __len__(self) -> int:  # Returns the size of the dataset
        return len(self.annotations)

    def __getitem__(self, index) -> Tuple[Image.Image, torch.Tensor]:
        """
        Get a single image and its corresponding label matrix.

        Args:
            index (int): The index of the image in the csv dataframe list.
        Returns:
            tuple ( torch.Tensor, tensor) :
                - (Single image, label bounding box matrix).
                - The image has been normalized.
                - The label tensor is of shape (S, S, CELL_NODES).
        """
        cfg = self.cfg
        S, B, C = cfg.S, cfg.B, cfg.C

        # --- 1 Make sure the index doesn't go out of bounds of the dataframe. However when self.num_samples == 0 that means grab entire dataset
        if index >= self.num_samples and self.num_samples != 0:
            print(
                f"\n\nERROR: Index is out of bounds. Index: {index}, size of dataframe: {self.__len__()}, index starts at zero. \n\nOccurred at: VOCDataset.__getitem__()"
            )
            sys.exit(0)  # exit program

        # --- 2: Use the index to retrieve the image and its annotation's .xml from the dataframe to create their file paths.
        anno_xml_path = os.path.join(
            self.dataset_path, cfg.ANNOTATIONS_DIR_NAME, self.annotations.iloc[index, 1]
        )
        img_path = os.path.join(
            self.dataset_path, cfg.IMAGES_DIR_NAME, self.annotations.iloc[index, 0]
        )

        # --- 3: Extract Annotations.
        img, boxes = self.voc_extraction.load_sample(
            anno_xml_path=anno_xml_path, img_path=img_path
        )

        # --- 4: Create the label tensor.
        #    Note: the label and model output tensors will be the same shape. However the labels second bounding box will be zero-ed out.
        label_matrix = torch.zeros((S, S, B * 5 + C))  # (7, 7, 30)

        # --- 6: Add the annotations to the label tensor.
        for box in boxes:
            cls_idx, x, y, width, height = box.tolist()

            # --- 6.1: Find which cell the box's midpoint is in.
            i, j = int(S * y), int(S * x)
            # i, j represents (row, col) of the grid cells, locates the cell that contains the bbox's mid-point.

            # --- 6.2: Make the x and y coords relative to the grid cell that contains the box, instead of being relative to the entire image.
            x_rel, y_rel = S * x - j, S * y - i

            # # --- 6.3: Add the coordinates.
            if (
                label_matrix[i, j, C] == 0
            ):  # Make sure we haven't already added a box to this cell, this is position pc₁.
                # NOTE: One caveat of YOLOv1 is it has a difficult time predicting small objects. Example: if two bounding boxes are in the same cell then only one will be selected. This is less likely to occur if split_size is larger.
                label_matrix[i, j, C] = 1  # Set pc₁ to 1, i.e an object exists here.
                coords = torch.tensor([x_rel, y_rel, width, height])
                # Add the coordinates at the first bounding boxes position of that cell.
                label_matrix[i, j, C + 1 : C + 5] = coords
                # Add the class_idx
                label_matrix[i, j, int(cls_idx)] = 1

        return img, label_matrix


# Test run module:
#   $          python -m data.voc_dataset
def test():
    print("\n\n🚧 TESTING: voc_dataset module \n\n")

    cfg = load_config("config_voc_dataset.yaml")
    t = setup_transforms(cfg.IMAGE_SIZE)
    # d = VOCDataset(cfg, which_dataset=cfg.TRAIN_DIR_NAME, transforms=t)
    d = VOCDataset(
        cfg,
        which_dataset=cfg.VALIDATION_DIR_NAME,
        num_samples=cfg.NUM_VAL_SAMPLES,
        transforms=t,
        print_df=True,
    )
    # print(d.__len__())
    print(len(d))
    # image, label = d.__getitem__(1)
    # print(image.shape, label.shape)


if __name__ == "__main__":
    test()
