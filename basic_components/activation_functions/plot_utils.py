import matplotlib.pyplot as plt
import numpy as np

def plot_activations(input_values, activated_values, title:str, yLabel:str, legend_label:str):
    """
    Plots activation functions

    Args:
        input_values (tensor): The values before the activation has been applied.
        activated_values (tensor): The values after the activation has been applied.
        title (str): The title of the graph.
        yLabel (str): The label to give the y label.
        legend_label (str): The function formula in latex format.
            - Example: r'$\sigma(x) = \frac{1}{1 + e^{-x}}$'
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(input_values, activated_values, label=legend_label)

    # Add the line down the middle (horizontal line at y=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5)

    # Set the x-axis to pass through y=0
    ax.spines['left'].set_position('zero')

    # Hide the original top and right spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # Add labels, title, and legend
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel(yLabel)
    ax.legend()
    ax.grid(True)
