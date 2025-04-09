"""
Methods to convert bounding boxes from corner_coordinates VOC (Xmin, Ymin, Xmax, Ymax) to mid-point_coordinates YOLO (x, y, height, width) and vice versa.

Note: the face-mask dataset is in VOC format
"""

def voc_to_yolo(img_size, voc):
    """
    Convert PASCAL VOC format to YOLO format, from absolute coordinates "corner-points" [Xmin, Ymin, Xmax, Ymax] to YOLO "midpoint" [x_center, y_center, height, width].
    
    YOLO: the coordinates are normalized between 0 and 1, because data of the bounding box is the percentage of the image ex: [0.50991.., 0.51, 0.974..., 0.972], 0.50991 means midpoint center X point is located at 50% of the image on the x-axis.
    
    PASCAL VOC: absolute coordinates relative to the entire image [Xmin, Ymin, Xmax, Ymax]
    
    Parameters
    ----------
        img_size -> tuple(int, int): width, height of the image
        voc -> list(int, int, int, int, int, int, int): c1, c2, c3, x_min, y_min, x_max, y_max the corner-points.
    
    Returns
    -------
        List [int, int, int, float, float, float, float]: YOLO bounding boxes mid-point coordinates c1, c2, c3, x_center, y_center, width, height.
    """
    c1, c2, c3, x_min, y_min, x_max, y_max = voc
    img_width, img_height = img_size
    
    
    x_center = ( ( x_min + x_max ) / 2 ) / img_width
    y_center = ( ( y_min + y_max ) / 2 ) / img_height
    bbox_width = ( x_max - x_min ) / img_width
    bbox_height = ( y_max - y_min ) / img_height
    return [c1, c2, c3, x_center, y_center, bbox_width, bbox_height]


def yolo_to_voc(bbox, img_width, img_height):
    """
    Convert YOLO format (normalized) mid-point to VOC format (absolute pixels) corner-points.
    
    Parameters:
        bbox : list(int, int, int, float, float, float, float, float)
            x_center, y_center, width, height Normalized (0 to 1) YOLO mid-point coordinates of the bounding box
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
        
    Returns:
        tuple (int): (c1, c2, c3, xmin, ymin, xmax, ymax) in absolute pixel values
    """
    c1, c2, c3, x_center, y_center, width, height = bbox
    box_width = width * img_width
    box_height = height * img_height

    center_x = x_center * img_width
    center_y = y_center * img_height

    xmin = int(center_x - box_width / 2)
    xmax = int(center_x + box_width / 2)
    ymin = int(center_y - box_height / 2)
    ymax = int(center_y + box_height / 2)
    return c1, c2, c3, xmin, ymin, xmax, ymax
