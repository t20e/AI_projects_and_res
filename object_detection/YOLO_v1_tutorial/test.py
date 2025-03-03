import torchvision
from torchvision.io import read_image

# Replace with the path to your image
image_path = "./test.jpg"

# Load image as a PyTorch tensor
image = read_image(image_path)

print("Image loaded successfully!", image.shape)
