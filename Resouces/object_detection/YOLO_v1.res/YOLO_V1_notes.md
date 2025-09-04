# YOLO V1  Algorithm

TUTORIAL BY ALADDIN PERSSON | <a href="https://www.youtube.com/watch?v=n9_XyCGr-MI&t=2224s">Link</a>


<a href="https://arxiv.org/pdf/1506.02640">YOLO V1 PAPER</a>

<a href ="https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2">DATASET</a>



- The idea of YOLO is to look at an image and be able to plot bounding boxes around the objects in the image, and be able to tell what those objects are, example below is two dogs.


Input Image: <img src="./ref_imgs/input_image.png" alt="Description" width="400">

1. Then split the image into an SxS grid, example here is 3x3


Split Grid Image: <img src="./ref_imgs/split_grid_iamge.png" alt="Description" width="400">

2. Now we have 9 <span class="cell-text">cells</span>.



3. The idea is that each <span class="cell-text">cell</span> will output a prediction with a corresponding bounding box <span class="boundingBox-text">bounding box</span>.



4. We can see that each dog is in multiple cells, and so we want only one bounding box for each object, the idea is that we will find one <span class="cell-text">cell</span> that is responsible for that object.
    1. The <span class="cell-text">cell</span> that is responsible for outputing that objects bounding box will contain that objects <span class="midPoint-text">mid-point</span>.


        Mid-Point Example: <img src="./ref_imgs/mid_point_example.png" alt="Description" width="400">

    2. Each output and label will be relative to the <span class="cell-text">cell</span>!

        One Cell: <img src="./ref_imgs/one_cell.png" alt="Description" width="400">
        1. The X, Y are the coordinates for the objects <span class="midPoint-text">mid-point</span> in that cell. The X, Y values are going to be between 0 and 1.
        2. The W, H are going to be the width and height of that object relative to the cell. The W and H can be greater than 1, becuase the object can be larger than the <span class="cell-text">cell</span>.
        3. In this one <span class="cell-text">cell</span> example the coordinates for the <span class="midPoint-text">mid-point</span> could be <span class="highLight-text" >[0.95, 0.55, 0.5, 1.5]</span>. X = 0.95 Y = 0.55, the width of the object is 0.5 of a total cell, the height is 1.5, so its 1.5 times the size of the cell.

        Output Example:  <img src="./ref_imgs/output_example.png" alt="Description" width="400">


5. How the labels and Predictions will actually look?

    Labels look like:  <img src="./ref_imgs/howWillLabelsLook.png" alt="Description" width="400">

    1. 20 different classes of objects we are tring to predict in the dataset, Pc= probability that there is an object in that cell, and then the <span class="boundingBox-text">bounding boxes</span> of that object. Predictions will look very similar, but we will ouput two <span class="boundingBox-text">bounding boxes</span> instead of one, the idea: they will specialize to output different <span class="boundingBox-text">bounding boxes</span> (wide vs tall).s

    Prediction Example:  <img src="./ref_imgs/howWillPredictionsLookLike.png" alt="Description" width="400">

    2. <span class="highLight-text">NOTE:</span> One caveat to YOLO is that a <span class="cell-text">cell</span> can only detect one object. A way to get around this is to make the grid finer, meaning from 3x3 to 19x19 etc.. of course this will need more compute power.
    3. Pc<sub>1</sub> = probability that theres an object in first one predicted, Pc<sub>2</sub> = probability that theres an object in second one predicted, then the box1 and box2 of those coordinates.

    <a id="item-54"></a>
    <img src="./ref_imgs/remeberThisForEveryCell.png" alt="Description" width="400">
    
    4. <span  class="highLight-text">Remeber this for every cell</span>.
        1. Target shape: SxS= grid size, the 25 gets broken down into => 20 are for classes predictions + 1 is for the probability score + the remaining 4 are for the <span class="boundingBox-text">bounding boxes</span>.
        2. Prediction Shape: very similar to the Target shape, except we will have 1 more probability score and 4 additional <span class="boundingBox-text">bounding boxes</span> predictions. 25 + 1 + 4 = 30


6. **Understand The Model Architecture And Implement in PyTorch**


    ⭐ Check out Notes on [Architecture](../../../object_detection/yolo_v1_orig/README.md#yolo-v1-architecture).


    ⭐ [Architecture code implementation](../../../object_detection/yolo_v1_orig/model/yolov1.py).


7. **Loss Function**

   ⭐ Check out on [notes loss function](./loss_fn.md).


   ⭐ [Loss function Code implementation](../../../object_detection/yolo_v1_orig/model/loss.py).





<!-- CSS CODE BELOW -->
<style>
.cell-text { color: #FF8C42; font-weight: 500; }
</style>
<style>
.boundingBox-text { color: #6699CC; font-weight: 500; }
</style>
<style>
.midPoint-text { color: #A23E48; font-weight: 500; }
</style>
<style>
.highLight-text { color:rgb(215, 236, 111); font-weight: 500; }
</style>