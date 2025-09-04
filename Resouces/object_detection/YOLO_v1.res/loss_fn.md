# YOLO Loss Function

<img src="./ref_imgs/loss_formulas.png" alt="Loss formulas" width="500">

#### [Code Implementation](../../../object_detection/yolo_v1_orig/model/loss.py)

*Steps correlated to the numbers on the img above.*


1.  <strong style="color: Magenta;">Localization Loss:</strong> Pinpointing the object. First two lines.
    $$ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B}  \text{𝟙}_{i j}^{\text{obj}} [ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 ] $$
    
    - These functions *penalizes* the model for inaccuracies in predicting the bounding boxes's position and size.
    - *Center Coordinates (x, y):* The term $\color{Magenta}(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2$
        - Calculates the squared root error between the predicted center coordinates $\color{magenta}x_i, y_i$ of a bounding box and the ground truth coordinates $\color{Magenta}\hat{x}_i, \hat{y}_i$

    $$ + \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B}  \text{𝟙}_{i j}^{\text{obj}} [ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] $$
    
    - *Width and Height (w,h):* The term $\color{Magenta}(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2$
        - Measures the error in the predicted width and height of the bounding box. .
        - By taking the square root, the model becomes more sensitive to small errors in small bounding boxes, which is essential for detecting smaller objects accurately.

    - *The indicator Function ($\color{Magenta}\text{𝟙}_{i j}^{\text{obj}}$):* This is a conditional switch. It equals 1 only if the *j-th* bounding box predictor in grid cell *i* is "responsible" for detecting an object.
        -  The predictor with the highest Intersection over Union (IoU) with the ground truth bounding box is deemed responsible. This ensures that only one predictor per object is penalized for localization errors.

    - *The Weighting Parameter ($\color{Magenta}\lambda_{\text{coord}}$):* This hyperparameter gives more weight to the localization loss. 
        - The paper set $\color{Magenta}\lambda_{\text{coord}}$ = 5.


2. <strong style="color: Teal;">Confidence Loss:</strong> Is there an object, and how well does the box fit? The third and fourth lines.

    $$ + \sum_{i=0}^{S^2} \sum_{j=0}^{B}  \text{𝟙}_{i j}^{\text{obj}}  (C_i - \hat{C}_i)^2  $$
    $$ + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B}  \text{𝟙}_{i j}^{\text{obj}}  (C_i - \hat{C}_i)^2  $$
    - The confidence score has a dual meaning: it reflects the presence of an object in the bounding box and the accuracy of the bounding box's shape (as measured by IoU).

    - *Confidence score ($\color{Teal}\text{C}$):* The confidence score is defined as $P(Object) * IoU(pred, truth)$.
        - If no object is present in a cell, $P(Object)$ = 0, and the confidence score should be 0.
        - If an object is present, $P(Object)$ = 1, and the confidence score should ideally be equal to the IoU between the predicted box and the ground truth box.

    - *The confidence loss is calculated differently for cells that contain an object versus those that do not.*

    $$ + \sum_{i=0}^{S^2} \sum_{j=0}^{B}  \text{𝟙}_{i j}^{\text{obj}}  (C_i - \hat{C}_i)^2  $$

    - *For cells with an object ($\color{Teal}\text{𝟙}_{i j}^{\text{obj}}$):* The term $\color{Teal}(C_i - \hat{C}_i)^2$.
        - Penalizes the difference between the predicted confidence $\color{Teal}C_i$ and the ground truth $\color{Teal}\hat{C}_i$, which is the IoU of the responsible predictor.

    $$ + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B}  \text{𝟙}_{i j}^{\text{noobj}}  (C_i - \hat{C}_i)^2  $$
    - *For cells without an object ($\color{Teal}\text{𝟙}_{i j}^{\text{noobj}}$):* The term $\color{Teal}(C_i - \hat{C}_i)^2$ is also used here however the target confidence $\color{Teal}\hat{C}_i$ = 0.
        - Since most of the cells in an image do not contain an object, this part of the loss dominates. To counteract this imbalance, the authors introduced $\color{Teal}\lambda_{\text{noobj}}$ hyperparameter. 
            - The paper set $\color{Teal}\lambda_{\text{noobj}}$ = 0.5.
            - This down-weights the confidence loss for cells that are not supposed to detect an object. Which prevents the model from becoming overly confident in predicting background.


3. <strong style="color: Lime;">Classification Loss:</strong> What is the object? The last line.

    $$ + \sum_{i=0}^{S^2}  \text{𝟙}_{i}^{\text{obj}}  \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2 $$

    - This measure how well the model predicts the correct class for the detected object.

    - *Class Probabilities ($\color{Lime}\text{p}(\text{c})$):*
        - For each grid cell that contains an object $\color{Lime}\text{𝟙}_{i}^{\text{obj}}$ the model predicts a probability distribution over all possible classes.

    - *Sum-Squared Error for Classes:* The term $\color{Lime}\sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$
        - Calculates the squared error between the predicted class probabilities $\color{Lime}(p_i(c))$ and the true one-hot encoded class probabilities $\color{Lime} \hat{p}_i(c)$.
        - Note this loss is only calculated for the grid cell containing the object, not for each bounding box predictor.

**The Goal:**  
- **Improve bounding box coordinates**: By minimizing the localization loss.  
- **Distinguish Object from background**: By learning to predict high confidence scores for boxes containing object and low scores for those without. 
- **Enhance bounding box shapes**: By encouraging the confidence score to reflect the IoU.
- **Correctly Classify objects**: By minimizing the classification error.