import numpy as np
import matplotlib.pyplot as plt


def draw_image_cell_image(image_array):
    # Create a figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(3, 3))

    # Display the array as an image.
    ax.imshow(image_array, cmap='gray', interpolation='none')

    # Add purple lines to show each cell
    ax.set_xticks(np.arange(-0.5, image_array.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, image_array.shape[0], 1))
    ax.grid(color='m', linestyle='-', linewidth=1)

    # Remove the tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            # Determine text color based on the cell's background color
            text_color = 'black' if image_array[i, j] == 1 else 'white'
            
            # Place the number at the center of each cell.
            ax.text(j, i, image_array[i, j], ha='center', va='center', color=text_color, fontsize=25)
    # Display the plot.
    plt.show()