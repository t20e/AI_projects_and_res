
"""
Set up Compose  Object that will apply transformations to images and label tensors.
"""

import torchvision.transforms as T

# <------------- Transforms ------------->
class CustomCompose(object):
    """Apply a sequence of transforms safely on (image, bboxes)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes


class Resize(object):
    """Resize the image. No change needed for bboxes since they are normalized (0-1)."""

    def __init__(self, size):
        self.size = size  # (width, height) ex: (448,448)

    def __call__(self, img, bboxes):
        img = T.Resize(self.size)(img)
        return img, bboxes  # bboxes stay the same


class ToTensor(object):
    """Convert image to Tensor. Leave label bboxes as they are."""

    def __call__(self, img, bboxes):
        img = T.ToTensor()(img)  # Automatically normalize image between 0-1
        return img, bboxes


def setup_transforms(img_size):
    """
    Resize and convert Image data to tensors. Performs image normalization under the hood. Won't alter label tensors.
    """
    return CustomCompose(
        [
            Resize((img_size, img_size)),  # Resize image to e.g. 448x448
            ToTensor(),  # Convert image to tensor
        ]
    )