from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import string
from math import ceil
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker


class Table:
    def __init__(self, df: pd.DataFrame, title: str, rows_per_page: int = 24, scale_x: float = 1, scale_y: float = 1.5,
                 first_columns_to_color: int = 3,
                 header_facecolor: str = '#E61030', first_columns_facecolor: str = '#ADD8E6',
                 other_columns_facecolor: str = '#FFFFFF',
                 fig_size: Tuple[float, float] = (20, 11.7 * 0.9)):
        """
        Initialize Table object.

        :param df: DataFrame to display in table.
        :param title: Title of the table.
        :param rows_per_page: Number of rows per page.
        :param scale_x: Table scale on x-axis.
        :param scale_y: Table scale on y-axis.
        :param first_columns_to_color: Number of first columns to color differently.
        :param header_facecolor: Color of the header cells.
        :param first_columns_facecolor: Color of the first few column cells.
        :param other_columns_facecolor: Color of other column cells.
        :param fig_size: Size of the figure.
        """
        self.df = df
        self.title = title
        self.rows_per_page = rows_per_page
        self.annotations = []
        self._scale_x = scale_x
        self._scale_y = scale_y
        self._first_columns_to_color = first_columns_to_color
        self._header_facecolor = header_facecolor
        self._first_columns_facecolor = first_columns_facecolor
        self._other_columns_facecolor = other_columns_facecolor
        self._fig_size = fig_size

    @property
    def scale(self) -> Tuple[float, float]:
        """Returns the current scale of the table."""
        return self._scale_x, self._scale_y

    @scale.setter
    def scale(self, values: Tuple[float, float]):
        """Sets the scale of the table."""
        self._scale_x, self._scale_y = values

    @property
    def first_columns_to_color(self) -> int:
        """Returns the number of first columns to color differently."""
        return self._first_columns_to_color

    @first_columns_to_color.setter
    def first_columns_to_color(self, value: int):
        """Sets the number of first columns to color differently."""
        self._first_columns_to_color = value

    @property
    def header_facecolor(self) -> str:
        """Returns the color of the header cells."""
        return self._header_facecolor

    @header_facecolor.setter
    def header_facecolor(self, value: str):
        """Sets the color of the header cells."""
        self._header_facecolor = value

    @property
    def first_columns_facecolor(self) -> str:
        """Returns the color of the first few column cells."""
        return self._first_columns_facecolor

    @first_columns_facecolor.setter
    def first_columns_facecolor(self, value: str):
        """Sets the color of the first few column cells."""
        self._first_columns_facecolor = value

    @property
    def other_columns_facecolor(self) -> str:
        """Returns the color of other column cells."""
        return self._other_columns_facecolor

    @other_columns_facecolor.setter
    def other_columns_facecolor(self, value: str):
        """Sets the color of other column cells."""
        self._other_columns_facecolor = value

    @property
    def fig_size(self) -> Tuple[float, float]:
        """Returns the size of the figure."""
        return self._fig_size

    @fig_size.setter
    def fig_size(self, values: Tuple[float, float]):
        """Sets the size of the figure."""
        self._fig_size = values

    def _style_cells(self, table, ax, table_data):
        """
        Styles the cells in the table.

        :param table: Table to style.
        :param ax: Axes object.
        :param table_data: Data to populate the table.
        """
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(self.header_facecolor)
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                if col < self.first_columns_to_color:
                    cell.set_facecolor(self.first_columns_facecolor)
                else:
                    cell.set_facecolor(self.other_columns_facecolor)
                cell.set_text_props(color='black', ha='center', alpha=1)
            cell.set_edgecolor('none')

    def add_annotations(self, annotations: List[str]):
        """Add annotations to the table.
        :param annotations: List of annotation strings.
        """
        if isinstance(annotations, list):
            self.annotations.extend(annotations)
        else:
            self.annotations.append(annotations)

    def _add_annotations_to_figure(self, fig):
        """Add annotations to the figure.
        :param fig: Figure object.
        """
        for i, annotation in enumerate(self.annotations, start=1):
            fig.text(0.05, 0.05 - 0.03 * i, annotation, fontsize=10, transform=plt.gcf().transFigure)

    def setup_table(self, table, ax, table_data):
        """
        Setup table style and other configurations.
        """
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        # table.auto_set_column_width(.001)
        table.auto_set_column_width(col=list(range(len(self.df.columns))))

    def save(self, file_name: str):
        """Save the table to a file.
        :param file_name: Name of the file.
        """
        num_pages = ceil(len(self.df) / self.rows_per_page)
        with PdfPages(file_name) as pdf_pages:
            for page in range(num_pages):
                start = page * self.rows_per_page
                end = (page + 1) * self.rows_per_page
                fig, ax = plt.subplots(figsize=self.fig_size)
                ax.set_facecolor('#eafff5')
                ax.set_title(f'{self.title} (Page {page + 1}/{num_pages})')
                table_data = self.df.iloc[start:end]
                table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
                table.scale(*self.scale)
                self._style_cells(table, ax, table_data)
                self.setup_table(table, ax, table_data)

                self._add_annotations_to_figure(fig)
                ax.axis('off')
                pdf_pages.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    def show(self):
        """Display the table."""
        num_pages = ceil(len(self.df) / self.rows_per_page)
        for page in range(num_pages):
            start = page * self.rows_per_page
            end = (page + 1) * self.rows_per_page
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.set_facecolor('#eafff5')
            ax.set_title(f'{self.title} (Page {page + 1}/{num_pages})')
            table_data = self.df.iloc[start:end]
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
            table.scale(*self.scale)
            self.setup_table(table, ax, table_data)

            self._style_cells(table, ax, table_data)
            self._add_annotations_to_figure(fig)
            ax.axis('off')
            plt.show()


class OrangeTable(Table):
    def __init__(self, *args, **kwargs):
        """
        Initialize the OrangeTable class.

        :param args: Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def setup_table(self, table: plt.Axes.table, ax: plt.Axes, table_data: pd.DataFrame):
        """
        Setup table style and other configurations specifically for OrangeTable.

        :param table: The table to be styled.
        :param ax: The axes of the plot.
        :param table_data: The data to be represented in the table.
        """
        print("Setting up orange table...")
        print("Table data shape:", table_data.shape)
        super().setup_table(table, ax, table_data)  # Call parent setup_table
        print("Cell (0, 0) before color change:", table[23, 18], "Facecolor:", table[23, 18].get_facecolor())

        # Apply additional styling if columns are more than or equal to rows
        if table_data.shape[1] >= table_data.shape[0]:  # Check if columns >= rows
            num_rows = len(table_data) + 1  # account for header row in matplotlib table
            num_cols = len(self.df.columns)

            # Loop through cells below the diagonal and color them differently
            for i in range(num_cols - 1, -1, -1):
                for j in range(num_rows - 1, num_cols - i, -1):
                    print(f"Changing color for cell ({j}, {i})")
                    table[j, i].set_facecolor('lightgrey')
        print("", table[23, 18], "Facecolor:", table[23, 18].get_facecolor())


n_rows = 24
n_cols = 39

# Generate random 6-letter strings
random_strings = [''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=6)) for _ in range(n_rows)]

random_strings1 = [''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=6)) for _ in range(10)]

# Create DataFrame
df = pd.DataFrame()

df1 = pd.DataFrame()

# Add first 3 columns of 6-letter strings
for i in range(3):
    df[f'col{i + 1}'] = random_strings

# Add remaining columns of random integers
# Add remaining columns of random integers with 6 digits
for i in range(3, n_cols):
    df[f'col{i + 1}'] = np.random.randint(100000, 1000000, size=n_rows)

for i in range(3):
    df1[f'col{i + 1}'] = random_strings1

# Add remaining columns of random integers
# Add remaining columns of random integers with 6 digits
for i in range(3, n_cols):
    df1[f'col{i + 1}'] = np.random.randint(100000, 1000000, size=10)

# Create a table
table = Table(df1, "Table with 39 columns and 24 rows")
table.add_annotations("Testing the table annotations")
table.scale = (1, 1.7)
#table.first_columns_to_color = 0
table.save("table4.pdf")

table2 = OrangeTable(df, 'Orange Table')

# Change colors using setters
table2.header_color = mcolors.CSS4_COLORS['orangered']
table2.first_three_columns_color = mcolors.CSS4_COLORS['navajowhite']
table2.other_cell_color = mcolors.CSS4_COLORS['seashell']
table2.add_annotations("Testing Annotations")
table.first_columns_to_color = 3
table.scale = (1, 1.7)

# Save table to a PDF
table2.save("orange_table4_setters.pdf")


def random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

data = {
    'ShortString': [random_string(20) for _ in range(20)],
    'LongString': [random_string(200) for _ in range(20)]
}

df2 = pd.DataFrame(data)

df3 = df2.head(3)

table4 = Table(df2,"My Title")
table4.save("long_string.pdf")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


class TableWithSubplots(Table):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.set_tight_layout(True)
        self.gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.5, wspace=0.5)
 # setup GridSpec for 3 rows and 3 columns

    def show_table(self):
        ax_table = plt.subplot(self.gs[0, :])  # Table on the first row, spans all columns
        ax_table.axis('tight')
        ax_table.axis('off')
        ax_table.set_title(self.title)

        table_data = self.df
        table = ax_table.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
        self._style_cells(table, ax_table, table_data)

    def add_subplot(self, data, title, subplot_index):
        """Add a subplot with a line chart for a specific description."""
        row = (subplot_index // 3) + 1  # Calculate the row of the subplot
        col = subplot_index % 3  # Calculate the column of the subplot
        ax_plot = plt.subplot(self.gs[row, col])  # Subplots on the following rows
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.6, hspace=0.6)
        x = range(len(data.index))  # Indices for the x-axis
        print(f"X: {x}")


        y = data.values  # Transpose the data and extract the values for the y-axis
        print(f"Y:{y}")
        ax_plot.plot(x, y)
        ax_plot.set_title(title)
        plt.xticks(x, data.index, rotation=45)

        ymin = y.min()
        ymax = y.max()
        ax_plot.set_ylim(ymin, ymax)  # Set the y-axis limits based on the range of values

    # Set the y-axis limits based on the range of values

    # Set the y-axis limits based on the range of values

    # Set the y-axis limits based on the range of values

    # Set the y-axis limits based on the range of values

    # Set the y-axis limits based on the range of values

    def show(self):
        """Display the table and all subplots."""
        self.show_table()
        for i, (description, data) in enumerate(self.df.iterrows()):
            print(data[1:])
            self.add_subplot(data[1:], description, i)
        plt.tight_layout()  # Optional: Improve the spacing between subplots
        plt.show()  # Show all figures and subplots

    def save(self, filename):
        """Save the figure to a file."""
        self.show_table()
        for i, (description, data) in enumerate(self.df.iterrows()):
            self.add_subplot(data[1:], description, i)
        plt.tight_layout()  # Optional: Improve the spacing between subplots
        plt.savefig(filename, format='pdf')  # Save the figure in PDF format

np.random.seed(42)
data = np.random.randint(low=1e6, high=1e9, size=(6, 8))
description = ['desc'+str(i) for i in range(1, 7)]
columns = ['description', 't2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2022Q2', 't2022Q3', 't2022Q4']
df = pd.DataFrame(np.hstack((np.array(description)[:, np.newaxis], data)), columns=columns)

# Show and save the table and plots
table = TableWithSubplots(df, 'My Table and Plots')
table.show()
table.save('my_figure.pdf')

