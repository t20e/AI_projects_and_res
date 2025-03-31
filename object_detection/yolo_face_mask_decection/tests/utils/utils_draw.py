# TEST UTILS

from PIL import Image, ImageDraw, ImageFont
from IPython.display import display


def draw_x_y_axis(img, drawContext):
    width, height = img.size
        
    # Draw x-axis ticks and labels along the bottom
    for x in range(0, width, 50):
        drawContext.line([(x, height - 5), (x, height)], fill='white', width=1)
        drawContext.text((x + 2, height - 15), str(x), fill='white')

    # Draw y-axis ticks and labels along the left
    for y in range(0, height, 50):
        drawContext.line([(0, y), (5, y)], fill='white', width=1)
        drawContext.text((7, y - 7), str(y), fill='white')

    # Optional: Draw axis lines
    drawContext.line([(0, height - 1), (width, height - 1)], fill='white', width=1)  # x-axis
    drawContext.line([(0, 0), (0, height)], fill='white', width=1)  # y-axis

    # Display inline
    display(img)

def draw_bbox_on_img(img_path, bbox_coordinates):
    """
    Plots bboxes on images
    
        img_path (str): path to the image
        bbox_coordinates (list): [[without_mask|with_mask, X, Y, width, height]] list of bboxes on the image
    """
    img = Image.open(img_path).convert("RGB")
    drawContext = ImageDraw.Draw(img)

    for box in bbox_coordinates:
        mask, x1, y1, x2, y2 = box
        drawContext.rectangle([x1, y1, x2, y2], outline='white', width=2)

        # Add (x, y) label
        # label_text = f'({x}, {y})'
        label_text = f'{mask}'
        
        drawContext.text((x1 + 5, y1 - 15), label_text, fill='white')  # position label above the top-left corner

    draw_x_y_axis(img, drawContext) # draw the x-y axes