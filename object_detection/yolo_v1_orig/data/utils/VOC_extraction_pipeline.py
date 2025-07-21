from PIL import Image
import xml.etree.ElementTree as ET
from typing import List, Dict
import os
import torch
from typing import Optional, Tuple
import sys

# My modules imports
from configs.config_loader import YOLOConfig, load_config
from data.utils.setup_transforms import CustomCompose
from data.utils.setup_transforms import setup_transforms


class VOCAnnotationsExtraction:
    def __init__(self, cfg: YOLOConfig, transforms: Optional[CustomCompose]):
        """
        VOCAnnotationsExtraction handles the entire pipeline for extracting an image and its annotations from the VOC dataset.

        Args:
            cfg (YoloConfig): Project Configurations.
        """
        self.cfg = cfg
        self.transforms = transforms

    def load_sample(self, anno_xml_path, img_path) -> Tuple[Image.Image, torch.tensor]:
        """
        Extract a single image and its corresponding annotations.

        Args:
            anno_xml_path (str): Path to the annotation file.
            img_path (str): Path to the image file.

        Return:
            img (Image.Image): Normalized image tensor data.
            tensor: Shape: (N, 5)
                - Where N is the number of bounding boxes and 5 -> [class_idx, x, y, w, h].
                - Coordinates are in mid-point format of Percentage values (normalized).
        """

        # --- 1: Open Image
        img = Image.open(img_path)
        # --- 2: Get the corner-points bboxes from the annotations.
        #   cp_abs_annos is a Dict of annotations with absolute values in corner-points format.
        cp_abs_annos = self.extract_annotations(xml_path=anno_xml_path)

        # --- 3: Convert the absolute values from corner-points to mid-point.
        #   anno is a Dict of annotations with absolute values in mid-point format.
        mp_abs_annos = self.convert_abs_to_mid_points(cp_abs_annos)

        # --- 4: Convert Absolute values to percentage values.
        mp_annos = self.convert_to_percentage(mp_abs_annos)

        # --- 5: Convert the label and image lists to tensor and
        annos_t = self.create_tensor(mp_annos)  # (N, 5)
        # --- 6: Apply transforms.
        if self.transforms:
            img, annos_t = self.transforms(img=img, bboxes=annos_t)
        return img, annos_t

    def create_tensor(self, anno: Dict) -> torch.Tensor:
        """Takes dictionary and returns an tensor of  (N, 5), where N is the number of objects, and 5 -> [cls_idx, x, y, w, h]"""
        a = []
        for obj in anno["objects"]:
            a.append(
                [
                    obj["class_idx"],
                    obj["x"],
                    obj["y"],
                    obj["w"],
                    obj["h"],
                ]
            )
        return torch.tensor(a)

    def convert_to_percentage(self, anno: Dict):
        """
        Converts bounding box values (x, y, w, h) from being absolute pixel values to being normalized percentages relative to the image.
        Args:
            anno (Dict): Dict of the annotations in mid-point format.
        Returns:
            Dict: Same dict with values converted to percentages
        """
        img_width = anno["img_width"]
        img_height = anno["img_height"]

        # Convert to percentages
        for box in anno["objects"]:
            box["x"] = box["x"] / img_width
            box["y"] = box["y"] / img_height
            box["w"] = box["w"] / img_width
            box["h"] = box["h"] / img_height

        return anno

    def extract_annotations(self, xml_path: str) -> list[dict]:
        """
        Extract labeled bounding box annotations from a single .xml file.

        Args:
            xml_path (str): The xml file to extract annotations from.
                - NOTE: The bounding box format must be in corner-points -> (xmin, ymin, xmax, ymax).
        Returns:
            Dictionary: nested dictionaries containing bounding boxes.
                - { "img_width": int,
                    "img_height": int,
                    "objects": [{obj_name:"bird", "class_idx":int, "xmin":int,
                                 "ymin":int, "xmax":int, ymax:int}]}.
                - Bounding box values are in Absolute pixel values.
            # NOTE: its probably better to use TypeDict dictionaries if you want to keep this data.
        """

        # --- 1: Parse the xml file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_size = root.find("size")
        annos = {  # annos short for annotations
            "img_width": int(img_size.find("width").text),
            "img_height": int(img_size.find("height").text),
            "objects": [],
        }

        # --- 2: Loop through each object
        for obj in root.findall("object"):
            # Get the object's name and index.
            obj_name = obj.find("name").text
            # class_names.index("bird")
            class_idx = self.cfg.CLASS_NAMES.index(obj_name)
            # Get the bounding box.
            box = obj.find("bndbox")
            # Append it.
            annos["objects"].append(
                {
                    "obj_name": obj_name,
                    "class_idx": class_idx,
                    "xmin": int(float(box.find("xmin").text)),
                    "ymin": int(float(box.find("ymin").text)),
                    "xmax": int(float(box.find("xmax").text)),
                    "ymax": int(float(box.find("ymax").text)),
                }
            )
        return annos

    def convert_abs_to_mid_points(self, annos: Dict) -> Dict:
        """
        Convert corner-points **absolute values** coordinates to midpoints **absolute values**.

        Args:
            annos (Dict): Dictionaries of the annotations.
        Returns:
            Dict: Converted to midpoints ->
                    { "img_width": int, "img_height": int, "objects": [{obj_name:"bird",
                    "class_idx":int, "x":int, "y":int, "w":int, h:int}]}.
        """
        for obj in annos["objects"]:
            # --- Extract coordinates.
            xmin = obj.pop("xmin")
            ymin = obj.pop("ymin")
            xmax = obj.pop("xmax")
            ymax = obj.pop("ymax")
            # --- Calculate x, y, w, h mid-points.
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + (w / 2)
            y = ymin + (h / 2)
            # Add to dict
            obj.update(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

        return annos


# Test module with:
# $     python -m data.utils.VOC_extraction_pipeline
def test():
    print("\n\n🚧 TESTING: VOC_extraction_pipeline module \n\n")

    cfg = load_config("config_voc_dataset.yaml")
    cwd = os.getcwd()
    xml_path = f"{cwd}/datasets/VOC_2012_dataset/train/Annotations/2007_000027.xml"
    img_path = f"{cwd}/datasets/VOC_2012_dataset/train/JPEGImages/2007_000027.jpg"


    t = setup_transforms(cfg.IMAGE_SIZE)
    e = VOCAnnotationsExtraction(cfg=cfg, transforms=t)

    print("\nTesting load_sample()\n\n ")
    print(e.load_sample(anno_xml_path=xml_path, img_path=img_path))

    # print("\n\nTesting extract_annotations(), output: \n")
    # print(e.extract_annotations(xml_path=xml_path))

    # print("\n\nTesting convert_abs_to_mid_points(), output: \n")
    # a = {
    #     "img_width": 340,
    #     "img_height":124,
    #     "objects": [
    #         {
    #             "obj_name": "train",
    #             "class_idx": 13,
    #             "xmin": 263,
    #             "ymin": 32,
    #             "xmax": 500,
    #             "ymax": 295,
    #         },
    #         {
    #             "obj_name": "train",
    #             "class_idx": 13,
    #             "xmin": 100,
    #             "ymin": 200,
    #             "xmax": 300,
    #             "ymax": 400,
    #         },
    #     ]
    # }

    # print(e.convert_abs_to_mid_points(a))


if __name__ == "__main__":
    test()
