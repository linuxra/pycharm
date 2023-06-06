import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def create_advanced_gridspec_figure(rows, cols, width_ratios=None, height_ratios=None, wspace=None, hspace=None):
    """
    Creates a figure with a GridSpec layout and returns the Fig and Axes objects.

    :param rows: Number of rows in the grid
    :param cols: Number of columns in the grid
    :param width_ratios: List of width ratios for each column
    :param height_ratios: List of height ratios for each row
    :param wspace: Width space between subplots
    :param hspace: Height space between subplots
    :return: Fig and Axes objects with the GridSpec layout set
    """
    fig = plt.figure()

    if width_ratios is None:
        width_ratios = [1] * cols

    if height_ratios is None:
        height_ratios = [1] * rows

    gs = GridSpec(rows, cols, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)

    if wspace is not None:
        gs.update(wspace=wspace)

    if hspace is not None:
        gs.update(hspace=hspace)

    # Create a list to store the axes
    axes = []

    # Add subplots
    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    return fig, axes


# Example usage:
rows, cols = 2, 2
width_ratios, height_ratios = [2, 1], [1, 2]
wspace, hspace = 0.3, 0.3
fig, axes = create_advanced_gridspec_figure(rows, cols, width_ratios, height_ratios, wspace, hspace)

# Generate some sample data
x = np.linspace(0, 2 * np.pi, 100)

# Add plots to the axes
for i, ax in enumerate(axes):
    ax.plot(x, np.sin(x + i))

# Display the figure
plt.show()
