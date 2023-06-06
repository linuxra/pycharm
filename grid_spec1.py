import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def create_advanced_gridspec_figure(layout, wspace=None, hspace=None):
    """
    Creates a figure with a flexible GridSpec layout and returns the Fig and Axes objects.

    :type layout: tuple
    :param layout: A list of tuples, where each tuple represents a subplot and contains the row index,
                   column start index, and column end index. For example, (0, 0, 0) specifies a subplot
                   in the first row (0th index), starting and ending at the first column (0th index).
                   Thus, you can define any grid layout by specifying the row, start column, and end
                   column for each subplot in the layout list.
    :param wspace: Optional; Width space between subplots
    :param hspace: Optional; Height space between subplots
    :return: fig (matplotlib.figure.Figure object), axes (list of matplotlib.axes.Axes objects)
             with the GridSpec layout set
    """
    # Calculate the maximum number of rows and columns based on the layout
    rows = max([cell[0] for cell in layout]) + 1
    cols = max([cell[2] for cell in layout]) + 1

    fig = plt.figure()

    gs = GridSpec(rows, cols, figure=fig)

    if wspace is not None:
        gs.update(wspace=wspace)

    if hspace is not None:
        gs.update(hspace=hspace)

    axes = []

    for cell in layout:
        ax = fig.add_subplot(gs[cell[0], slice(cell[1], cell[2] + 1)])
        axes.append(ax)

    return fig, axes


# Example usage:
layout = [
    (0, 0, 0), (0, 1, 1),  # Two plots in the first row
    (1, 0, 0), (1, 1, 1), (1, 2, 2)  # Three plots in the second row
]
fig, axes = create_advanced_gridspec_figure(layout, wspace=0.3, hspace=0.3)

# Generate some sample data
x = np.linspace(0, 2 * np.pi, 100)

# Add plots to the axes
for i, ax in enumerate(axes):
    ax.plot(x, np.sin(x + i))

# Display the figure
plt.show()
