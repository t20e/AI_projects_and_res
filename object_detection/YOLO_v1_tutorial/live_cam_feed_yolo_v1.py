"""

Live Cam-Feed YOLO-V1

NOTE:  I could not get it to work properly on live-feed cam. It would correctly identify the object from the web cam feed, but it would not draw the bounding box correctly around the object, however when used a image from the training dataset frame = cv2.imread('./data/images/000034.jpg'), it would correctly draw the bounding box around the trains from image. This is most likely because the model is overfitting to the training dataset.


"""
import matplotlib.patches as patches
import torchvision.transforms as transforms

import torch 
from model import Yolov1
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    save_model,
    load_model #my loader below
)
from loss import YoloLoss
import cv2
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms as T
from PIL import Image



DEVICE = torch.device('mps')# to use the M1 MAC GPU
NUM_WORKERS = 2
LEARNING_RATE = 2e-5


S = 7 # split size how many cells are we spliting the image into 7x7 = 49 cells
num_boxes = 2 # number of boxes that each cell predicts
C = 20 # num of classes
# List of 20 Pascal VOC classes (YOLO v1 trained on Pascal VOC dataset)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


###Load model
# model = load_model(
#     "a1-4096-100-images-to0-0.90-mAP",
#     Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE),
#     DEVICE
# )

# LOAD_MODEL_FILE = "./saved_models/overfitted-YoloV1-train-on-100img-and-4096.pt" 
LOAD_MODEL_FILE = "./saved_models/YoloV1-train-on-entire-dataset-4096-03_23_25.pt" 

model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
# model.eval()
 

def draw_bboxes_on_feed(frame, bboxes, frame_shape):
    """
    Draw Bounding boxes and labels on the web cam frame. The model takes a 448,448,3 but the orignal web cam will be a different size so after passing the resized image to the model, the output will be draw on the orignal web cam size
        
    Note:
        boxes format: [class_id, confidence_score, x1, y1, x2, y2]
        x1, y1 is the midpoint of the predicted bounding box relative to the entire image
        x2, y2 is the height and width of the bbox
        
    Parameters
    ----------
        frame (cv2-frame) 
        bboxes (list : list of the predicted bboxes that have gone through a NMS, and 
        frame_shape (tuple(int)) : orignal height & width of image (h, w)
    """
    orig_h, orig_w = frame_shape
    
    input_height, input_width = 448, 448
    # get scale rate
    scale_x = orig_w / input_width
    scale_y = orig_h / input_height
    
    
    fig, ax = plt.subplots(1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    ax.imshow(frame_rgb)
    
    for box in bboxes:
        class_id, conf_score, x1, y1, x2, y2 = box
        upper_left_x = x1 - x2 / 2
        upper_left_y = y1 - y2 / 2
        
        rect = patches.Rectangle(
            (upper_left_x * input_width * scale_x, upper_left_y * input_height * scale_y),
            x2 * input_width * scale_x,
            y2 * input_height * scale_y,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        
    plt.axis('off')  # Optional: turn off axis
    plt.title('CV2 Frame in Matplotlib')
    plt.show()
    
    
    
    
    

    # for box in bboxes:
    #     class_id, conf_score, x1, y1, x2, y2 = box
    #     print( x1, y1, x2, y2 )
        
        
        # convert the scale from 448, 448 to the orignal frame size
        # x1 = x1 * scale_x
        # y1 = y1 * scale_y
        # x2 = x2 * scale_x
        # y2 = y2 * scale_y
        
        # # Compute the top-left-corner (x,y) and the right-bottom-corner (x,y) so cv2 can draw it portional to the orignal image/frame by scale_x, scale_y
        # top_left_x = int((x1 - (y2 / 2)) * img_width * scale_x)
        # top_left_y = int((y1 - (x2 / 2)) * img_height * scale_y)
        # bottom_right_x = int((x1 + (y2 / 2)) * img_width * scale_x)
        # bottom_right_y = int((y1 + (x2 / 2)) * img_height * scale_y)
        # print(
        #     top_left_x,
        #     top_left_y,
        #     bottom_right_x,
        #     bottom_right_y,
        # )
        
        # print(f"Object : id:{class_id}, {VOC_CLASSES[int(class_id)], {conf_score}}")
        
        # # Draw the rectangle
        # cv2.rectangle(
        #     frame,
        #     (top_left_x, top_left_y),
        #     (bottom_right_x, bottom_right_y),
        #     (255, 255, 255), 2
        # )
        
        # # # Put class label
        # label = f"{VOC_CLASSES[int(class_id)]}: {conf_score:.2f}"
        # cv2.putText(frame, label, (top_left_x, top_left_y - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

class Compose(object):
    """Resize and transform to tensor"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes): # bboxes = bounding boxes
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

# one thing that can improve this is that u do a normalization. mean = 0 and standard deviation = 1
transforms = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor() # convert to tensors and NOTE: normalizes it under the hood
])


def run_inference_on_webcam():
    """
    Runs YOLO v1 model on live webcam feed
    
        Note: resizing the cv2 frame to be (448, 448) makes it unproportional, so resize a copy of the frane and then pass it to the model, and then redraw it proportionally to the orignal frame
    """
    cap = cv2.VideoCapture(0)
    # frame = cv2.imread('./data/images/000034.jpg') 

    
    test_counter = 0
    try:
        print("\n\n" + ("#" * 50))   # Press 'q' to exit
        print("Press Q to quit webcam!\n\n")
        
        if not cap.isOpened():
            print("\n\nError: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing webcam...")
                break
            # convert to RGB and transfer to PIL 
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Apply transform => resize and normalize
            model_input = transforms(image_pil, [])[0]
            # Add a batch to get shape (1, 3, 448, 448)
            model_input = model_input.unsqueeze(0).to(DEVICE)

            
            # # run interference
            with torch.no_grad():
                output = model(model_input) 
                # ouput is of shape torch.Size([1, 1470])

            bboxes = cellboxes_to_boxes(output)
            # the bboxes[0] below is becuase we are only working with one image at a time, if u have a batch loop thru it and pass its idx instead
            bboxes = non_max_suppression(bboxes[0], iou_threshold=0.6, threshold=0.5, box_format="midpoint")
            
            # img = cv2.resize(frame, (448, 448))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plot_image(img, bboxes)
            plot_image(model_input.squeeze(0).permute(1, 2, 0), bboxes)
            
            # draw_bboxes_on_feed(frame, bboxes, frame.shape[:2])
            
            # cv2.imshow("YOLOv1 Live Detection", frame)
            
            
            if test_counter > 20: 
                break;
            test_counter+=1


    finally:
        # Release the camera and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        print("\n\nWebcam and window closed.")


if __name__ == "__main__":
    run_inference_on_webcam()