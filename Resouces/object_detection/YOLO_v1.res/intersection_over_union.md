# IoU | Intersection Over Union


Prerequisite: [Understand Coordinate Formats](../understand_corner_and_mid_points.md)

**Implementation:** See [YOLO_v1_orig](../../../object_detection/yolo_v1_orig/utils/IoU.py) or [IOU.py](./utils/IOU.py) and [IOU-test.py](./utils/IOU-test.py)


**💡 Question: How do we measure how good a bounding box is?**

**Intersection over Union (IoU)** is a metric used to measure how much a predicted bounding box overlaps with a ground truth (target) bounding box. It's a fundamental tool for evaluating the accuracy of object detectors.
- The IoU score ranges from 0 to 1:
- A score of 0 means no overlap.
- A score of 1 means perfect overlap.
- The higher the IoU score, the more accurate the prediction's localization.

--- 

<img src="./ref_imgs/IOU_09.png" alt="Description" width="400"/>


**Calculate the Area of Intersection**


1. To find its area, we first need the coordinates of the intersection box itself (the yellow shaded box). Assuming we have corner points $(x_1, y_1, x_2, y_2)$ for both boxes:

    <img src="./ref_imgs/IOU_02.png" alt="Description" width="400"/>

    - Intersection box's top-left corner $(x_1, y_1)$:
        - $x_1 = \max(\text{pred\_box\_x1}, \text{true\_box\_x1})$
        - $y_1 = \max(\text{pred\_box\_y1}, \text{true\_box\_y1})$

    - Intersection box's bottom-right corner $(x_2, y_2)$:
        - $x_2 = \min(\text{pred\_box\_x2}, \text{true\_box\_x2})$
        - $y_2 = \min(\text{pred\_box\_y2}, \text{true\_box\_y2})$

    - The area is then calculated from these new coordinates.
        - If $x_1 >= x_2 \quad or \quad y_1 >= y_2$ the boxes don't overlap, and the intersection area is 0.

2. Calculate the Area of Union
    - The union is the total area covered by both boxes, highlighted here in pink.

    <img src="./ref_imgs/IOU_03.png" alt="Description" width="400"/>

    - To avoid double-counting the overlapping area, we use the following formula:

        - $\text{Union} = \text{Predicted Box Area} + \text{True Box Area} − \text{Intersection}$


3. Calculate the IoU by plugging in the area values.

$$ IOU = \frac{\text{intersection}}{\text{union}}$$



