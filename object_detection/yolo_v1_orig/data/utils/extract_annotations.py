import xml.etree.ElementTree as ET
import os
import pandas as pd  #
from typing import List, Dict
from configs.config_loader import load_config, YOLOConfig

def extract_annotations(cfg:YOLOConfig, xml_path:str) -> list[dict]:
    """
    Extract labeled bounding box annotations from a single xml file.

    Args:
        xml_path (str): The xml file to extract annotations from.
            - NOTE: The bounding box format must be in corner-points -> (xmin, ymin, xmax, ymax).
    returns:
        Array: nested dictionaries containing bounding boxes.
            - [{obj_name:"bird", class_idx:0, xmin:12, ymin:12, xmax:16, ymax:10}].
    """
    bboxes = []

    # --- 1: Parse the xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- 2: Loop through each object
    for obj in root.findall("object"):
        # Get the object's name and index.
        obj_name = obj.find("name").text

        # class_names.index("bird")
        class_idx = cfg.CLASS_NAMES.index(obj_name)

        # Get the bounding box.
        box = obj.find("bndbox")

        # Append it.
        bboxes.append(
            {
                "obj_name":obj_name,
                "class_idx": class_idx,
                "xmin": int(box.find("xmin").text),
                "ymin": int(box.find("ymin").text),
                "xmax": int(box.find("xmax").text),
                "ymax": int(box.find("ymax").text),
            }
        )
    return bboxes

def test():
    cfg = load_config("yolov1.yaml")
    cwd = os.getcwd()
    xml_path = f"{cwd}/datasets/VOC2012_train_val/Annotations/2007_000042.xml"
    print(extract_annotations(cfg=cfg, xml_path=xml_path))


# test()
