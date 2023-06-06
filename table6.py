import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import six
np.random.seed(42)
data = np.random.randint(low=int(1e6), high=int(1e9), size=(6, 8))
description = ['desc'+str(i) for i in range(1, 7)]
columns = ['description', 't2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2022Q2', 't2022Q3', 't2022Q4']
df = pd.DataFrame(data, columns=columns[1:])
df.insert(0, 'description', description)

# Transpose the dataframe
df_T = df.set_index('description').T
# Define a function to smooth data
def smooth_data(x, y):
    # Create a cubic spline interpolation
    interp = interp1d(x, y, kind='cubic')
    # Use more points for a smoother plot
    xnew = np.linspace(x.min(), x.max(), 500)
    ynew = interp(xnew)
    return xnew, ynew

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if k[1] == 0:  # Add this condition to check if the cell is in the first column
                cell.set_facecolor('#ADD8E6')
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    ax.set_title('Data Table', pad=20)
    return ax

render_mpl_table(df, header_columns=0, col_width=2.0)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot each series
for i, column in enumerate(df_T.columns):
    ax = axs[i//3, i%3]
    y = df_T[column].values
    x = np.arange(len(y))
    xnew, ynew = smooth_data(x, y)
    ax.plot(xnew, ynew, label=column)
    ax.set_title(column)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Values')

fig.suptitle('My Common Title', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.show()