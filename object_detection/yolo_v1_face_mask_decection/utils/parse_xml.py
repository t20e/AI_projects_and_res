# Parse xml VOC dataset labels
import xml.etree.ElementTree as ET


def parse_voc_annotation(xml_file_path, num_classes):
    
    """
    Parses xml annotations for one image for the dataset
        xml (str): path to xml file
    
    returns (python.list);
        list:
            for every bounding box in that image, [[0, 1, 0, 148, 75, 201, 133], [1, 0, 1 27, 78, 56, 106], etc..]
            [c1, c2, c3, x1, y1, x2, y2]]
    """
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    bboxes = []
    get_class_idx = lambda s: 0 if s.upper() == "WITH_MASK" else 1 if s.upper() == "WITHOUT_MASK" else 2 if s.upper() == "MASK_WEARED_INCORRECT" else None
    # None is worst case miss-labeled mask type, will cause error when creating tensor

    for obj in root.findall('object'):
        label = obj.find('name').text
        # print(label)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        classes_arr = [0] * num_classes
        classes_arr[get_class_idx(label)] += 1
        classes_arr.extend( [xmin, ymin, xmax, ymax])
        bboxes.append( classes_arr)
    return bboxes