"""
Utils to convert coordinates from mid-point to corner-points and vice versa.
"""

from typing import List, Dict


def convert_to_mid_points(a: List[Dict]):
    """
    Convert corner-points to midpoints.

    Args:
        a (List): List of bboxes in dictionaries ->  [{obj_name:"bird", class_idx:0, xmin:12, ymin:12, xmax:16, ymax:10}]
    Returns:
        List: same list but the coordinates converted to midpoints.
    """
    for box in a:
        # Extract coordinates
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']
        # Calculate x, y, w, h
        w = xmax - xmin # The width is the difference between the x from the bottom-right corner and the x from the top-left corner.
        h = ymax - ymin # Vice versa for the height.
        
        print(w)

def test():
    a = [
        {
            "obj_name": "train",
            "class_idx": 13,
            "xmin": 263,
            "ymin": 32,
            "xmax": 500,
            "ymax": 295,
        },
        {
            "obj_name": "train",
            "class_idx": 13,
            "xmin": 1,
            "ymin": 36,
            "xmax": 235,
            "ymax": 299,
        },
    ]
    convert_to_mid_points(a)

test()
