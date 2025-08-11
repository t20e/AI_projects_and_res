"""Plot losses"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def plot_losses(loss_history: dict[str, list[int]]):
    """
    Plot losses.

    Args:
        all_losses (dict):
            loss_history = {
                "mean_loss": [],
                "coord_loss": [],
                "conf_obj_loss": [],
                "noobj_loss": [],
                "class_loss": [],
                "mAP": [],
                "lr": [],
            }
    """

    # Create a figure and axis for the plot
    plt.figure(figsize=(10, 6))

    colors = ["b", "r", "g", "c", "m", "y", "k"]
    ax = plt.gca()

    # Iterate through the dictionary to plot each loss curve
    for idx, (loss_type, l) in enumerate(loss_history.items()):
        epochs = np.arange(1, len(l) + 1)
        color = colors[idx % len(colors)]
        # Plot the losses
        ax.plot(
            epochs, np.array(l), marker="o", linestyle="-", color=color, label=loss_type
        )
        print(f"\n, {loss_type}:, {l}")

    ax.set_title("Loss over Epochs", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (Log scale)", fontsize=12)

    ax.grid(True)
    ax.legend()

    # Set y-axis to a logarithmic scale
    ax.set_yscale("log")

    # Display the log scale as float values instead of e.g: 10^-4
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))

    # Show the plot
    plt.show()


# test as module
# $    python -m utils.plot.plot_loss
def test():
    loss_history = {
        "mean_loss": np.random.rand(50) * 0.01,
        "coord_loss": np.random.rand(50) * 0.120,
        "conf_obj_loss": np.random.rand(50) * 0.8,
        "noobj_loss": np.random.rand(50) * 0.07,
        "class_loss": np.random.rand(50) * 00.1,
        "mAP": np.random.rand(50) * 00.1,
    }
    plot_losses(loss_history)


if __name__ == "__main__":
    test()
