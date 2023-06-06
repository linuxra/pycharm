
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def visualize_dataframe_to_pdf1(df, title, filename, figsize=(26, 6)):
    # Set up the subplot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Define header color and text color
    header_color = mcolors.CSS4_COLORS['steelblue']
    header_text_color = 'black'
    fig.suptitle(title, fontsize=16, y=0.95)

    # Get all the columns in the dataframe
    subset_columns = df.columns

    # Create cell colors array
    cell_colors = np.full((df.shape[0] + 1, len(subset_columns)), 'white')
    cell_colors[0, :] = header_color

    # Create cell text colors array
    cell_text_colors = np.full((df.shape[0] + 1, len(subset_columns)), 'black')
    cell_text_colors[0, :] = header_text_color

    # Create the table
    table_width = 0.9  # 80% of the figure width
    table = ax.table(cellText=np.vstack([subset_columns, df[subset_columns].values]), cellLoc='center', loc='center',
                     cellColours=cell_colors,
                     bbox=[0.5 - table_width / 2, 0.5 - table_width / 2, table_width, table_width])
    table.auto_set_font_size(False)
    table.auto_set_column_width(False)
    for cell_coordinates, cell in table._cells.items():
        r, c = cell_coordinates
        cell.set_text_props(weight='bold' if r == 0 else 'normal', color=cell_text_colors[r, c],
                            fontsize=6 if r == 0 else 9)
        cell.set_linewidth(0.01)

    # Set the column width

    ax.axis('off')

    # Tighten layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00, top=0.96)

    # Save the plot to a PDF file
    with PdfPages(filename) as pdf:
        pdf.savefig(fig)

    # Close the figure to release memory
    plt.close(fig)


num_rows = 10
num_columns = 40
low = 100_000
high = 1_000_000

# Generate random six-digit integers
data = np.random.randint(low, high, size=(num_rows, num_columns))

# Create column names
column_names = [f'Column{i + 1}' for i in range(num_columns)]

# Create the DataFrame
df = pd.DataFrame(data, columns=column_names)
# visualize_dataframe_to_html2(df,title="RAI Data for past 36 months", filename='output2.html')
visualize_dataframe_to_pdf1(df, title="RAI Data for past 36 months", filename='output2.pdf')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
