import numpy as np
import matplotlib.pyplot as plt


def draw_image_cell_image(image_array):
    # Create a figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(3, 3))

    # Display the array as an image.
    ax.imshow(image_array, cmap="gray", interpolation="none")

    # Add purple lines to show each cell
    ax.set_xticks(np.arange(-0.5, image_array.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, image_array.shape[0], 1))
    ax.grid(color="m", linestyle="-", linewidth=1)

    # Remove the tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            # Determine text color based on the cell's background color
            text_color = "black" if image_array[i, j] == 1 else "white"

            # Place the number at the center of each cell.
            ax.text(
                j,
                i,
                image_array[i, j],
                ha="center",
                va="center",
                color=text_color,
                fontsize=25,
            )
    # Display the plot.
    plt.show()


def create_black_white_image_dataset(num_samples: int):
    """
    Create a dataset of 3x3 images and labels. The images only containing black and white pixels. An image's label is 1 if the image contains more white pixels than black and 0 otherwise.

    Args:
        num_samples: How many images to make.

    Return:
        Tuple (images, labels)
    """
    images = np.random.randint(2, size=(num_samples, 3, 3))
    white_pixel_counts = np.sum(images, axis=(1, 2))
    labels = (white_pixel_counts >= 5).astype(int)
    return images, labels
