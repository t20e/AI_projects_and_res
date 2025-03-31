from .utils_draw import draw_bbox_on_img, draw_x_y_axis



import xml.etree.ElementTree as ET

def parse_voc_annotation(xml_file_path):
    
    """
    Parses xml annotations for one image for the dataset
        xml (str): path to xml file
    
    returns (list);
        list for every bounding box in that image, [[-1, 148, 75, 201, 133], [0, 27, 78, 56, 106]]
        [[mask_value, x1, y1, x2, y2]]
    """
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    bboxes = []
    mask_value = lambda s: 1 if s.upper() == "WITH_MASK" else 0 if s.upper() == "WITHOUT_MASK" else -1 if s.upper() == "MASK_WEARED_INCORRECT" else None
    # None is worst case miss-labeled mask type, will cause error when creating tensor

    for obj in root.findall('object'):
        label = obj.find('name').text
        # print(label)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([mask_value(label), xmin, ymin, xmax, ymax])

    return bboxes