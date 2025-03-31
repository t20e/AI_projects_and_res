from PIL import Image, ImageDraw,ImageFont
# from IPython.display import display
import torchvision.transforms as T
from .convert_coordinates import yolo_to_voc

def plot_x_y_axis(img, drawContext):
    """
    Plots the X, Y axes on the PIL ImageDraw 
    """
    width, height = img.size
        
    # Draw x-axis ticks and labels along the bottom
    for x in range(0, width, 50):
        drawContext.line([(x, height - 5), (x, height)], fill='white', width=1)
        drawContext.text((x + 2, height - 15), str(f"{x}x"), fill='white')

    # Draw y-axis ticks and labels along the left
    for y in range(0, height, 50):
        drawContext.line([(0, y), (5, y)], fill='white', width=1)
        drawContext.text((7, y - 7), str(f"{y}y"), fill='white')

    # Optional: Draw axis lines
    drawContext.line([(0, height - 1), (width, height - 1)], fill='white', width=1)  # x-axis
    drawContext.line([(0, 0), (0, height)], fill='white', width=1)  # y-axis

    return img

def plot_bbox_on_img(img_tensor, bbox_coordinates):
    """
    Plots bboxes on a single image
    
        img_tensor : (tensor)
            Image tensor, this should be the resized image that we will pass to the model
        bbox_coordinates : (tensor)
             [class_id, X, Y, width, height]] in YOLO format, the coordinates have to be relative to the entire image, not a single cell.
    """
    
    # Convert from tensor to PIL image
    to_pil = T.ToPILImage()
    img = to_pil(img_tensor).convert("RGBA")
    img_w, img_h = img.size
    drawContext = ImageDraw.Draw(img)
    
    if len(bbox_coordinates.tolist()) > 0:
        print(f"\n\nPlotting the bounding boxes on the image, there should be {len(bbox_coordinates.tolist())}, with_mask, without_mask, mask_worn_incorrectly")  
    
    for box in bbox_coordinates.tolist():     
        c1, c2, c3, x1, y1, x2, y2 = yolo_to_voc(box, img_w, img_h)
        drawContext.rectangle([x1, y1, x2, y2], outline='white', width=2)

        # Add (x, y) label
        classes = ["with_mask", "without_mask", "mask_worn_incorrectly"]
        idx = [c1, c2, c3].index(1) # only one of the classes should be 1
        label_text = f'{classes[idx]}'
        
        drawContext.text((x1 , y1-15), label_text, fill='white')  # position label above the top-left corner

    plot_x_y_axis(img, drawContext).show() # draw the x-y axes and plot
    # Display inline
    # display(img) # works for jupiter notebook only

def visualize_grid_on_img(img_tensor, cells=None, split_size=7):
    """
    Draws a grid on an image and optionally fills-in a specific cell.

    Parameters
    ----------
        image_path : (pytorch.tensor) shape [Channels, Height, Width]
            image matrix
        cells : list of tuple of (row, col), optional
            Cells to fill-in, use to visualize which cell contains the mid-point of an bounding box.
        split_size : int
            Number of grid splits (e.g., 7 for YOLOv1).
            
    Note: the grid start counting from 0.
    """
    
    # Convert from tensor to PIL image
    to_pil = T.ToPILImage()
    img = to_pil(img_tensor).convert("RGBA")
    
    # img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    img_width, img_height = img.size
    cell_width = img_width / split_size
    cell_height = img_height / split_size

    if cells:
        print(f"\n\nPlotting cells, the blue filled-in cells are the cells that contain the mid-points of a bounding box, there should be {len(cells)}")
        for cell in cells:
            row, col = cell
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 255, 128))
            draw.text((x1 , y1), f"{cell}", fill='white')  # position label above the top-left corner

    # Draw grid lines on the overlay
    for i in range(1, split_size):
        x = i * cell_width
        draw.line([(x, 0), (x, img_height)], fill=(255, 0, 0, 255), width=1)
    for i in range(1, split_size):
        y = i * cell_height
        draw.line([(0, y), (img_width, y)], fill=(255, 0, 0, 255), width=1)

    # Combine original image with overlay
    plot_x_y_axis(img, draw)
    result = Image.alpha_composite(img, overlay)
    result.show()