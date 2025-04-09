"""Non-Max Suppression"""

from .intersection_over_union import intersection_over_union
import torch

def non_max_suppression(
    bboxes, 
    IOU_threshold,
    min_threshold):
    """ 
    Performs NMS
    
    Note:
        Non-Maximum Suppression (NMS) is a vital post-processing step in many computer vision tasks, particularly in object detection. It is used to refine the output of object detection models by eliminating redundant bounding boxes and ensuring that each object is detected only once.
    
    Parameters:
        bboxes (python:list) : predicted bounding boxes [ [1, 0.9, x1, y1, x2, y2], [etc..], [etc..], etc..]
            the 1 represents the class id, example: 1 means its a car
            0.9 represents the probability
        
        IOU_threshold (float) : the iou threshold when comparing bounding boxes for NMS
        
        min_threshold (float) : the threshold to remove bounding boxes with a low confidence score
    """
    
    assert type(bboxes) == list
    
    # remove bounding boxes with a low confidence score
    bboxes = [box for box in bboxes if box[1] > min_threshold]
    
    # sort the bboxes with the highesst probability at the beginning
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    
    bboxes_after_nms = []
    
    while bboxes:
        # grab a box from queue
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
                if box[0] != chosen_box[0] # check to see if the classes are the same if the bbox classes are different than we dont want to compare them IOU is only done when comparing bboxes for the same class, example : a car and a horse bbox
                or intersection_over_union(
                    torch.tensor(chosen_box[2:]), # just pass the coordinates from chosen box (x1, y1, x2, y2)
                    torch.tensor(box[2:]),
                )
                < IOU_threshold # if the IOU is less than the threshold then we will keep that box
        ]
        
        bboxes_after_nms.append(chosen_box)
        
    return bboxes_after_nms
        