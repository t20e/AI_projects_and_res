# Face Wask detection With YOLO V1 Paper Model

* More notes and examples in the YOLO_v1_tutorial and in my res>ai repo

<a href="https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?select=images">Dataset</a>


### Process
1. Get the dataset in YOLO format.
2. Convert the true bboxes to be relative in size to the cell containing its mid-point.
3. Pass the data to the model
4. Convert the ouput bboxes to be relative to the entire image.

