import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
from string import ascii_letters, digits
import string
import random
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as patches


from math import ceil

class Table:
    def __init__(self, df, title, rows_per_page=24):
        self.df = df
        self.title = title
        self.rows_per_page = rows_per_page
        self.annotations = []

    def _style_cells(self, table, ax,table_data):
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('lightseagreen')
                cell.set_text_props(color='white', fontsize=14,ha='center')
            else:
                cell.set_facecolor('lightcyan')
                cell.set_text_props(color='midnightblue',ha='center',alpha=1)
            cell.set_edgecolor('none')

    def setup_table(self, table,table_data):
        table.set_fontsize(14)
        # table.auto_set_column_width(list(range(len(self.df.columns))))

    def add_annotations(self, annotations):
        if isinstance(annotations, list):
            self.annotations.extend(annotations)
        else:
            self.annotations.append(annotations)

    def _add_annotations_to_figure(self, fig):
        for i, annotation in enumerate(self.annotations, start=1):
            fig.text(0.05, 0.05 - 0.03 * i, annotation, fontsize=10, transform=plt.gcf().transFigure)

    def save(self, file_name):
        num_pages = ceil(len(self.df) / self.rows_per_page)
        with PdfPages(file_name) as pdf_pages:
            for page in range(num_pages):
                start = page * self.rows_per_page
                end = (page + 1) * self.rows_per_page
                fig, ax = plt.subplots(facecolor='grey',figsize=(20, 6))
                ax.set_facecolor('#eafff5')
                ax.set_title(f'{self.title} (Page {page+1}/{num_pages})')
                table_data = self.df.iloc[start:end]
                table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
                self._style_cells(table, ax,table_data)
                self.setup_table(table, table_data)
                self._add_annotations_to_figure(fig)
                # for spine in ax.spines.values():
                #     spine.set_visible(False)
                #
                # # Create a rectangle patch with text just below the table
                # rect = patches.Rectangle((0, -0.1), 1, 0.1, transform=ax.transAxes, facecolor='red', clip_on=False)
                # ax.add_patch(rect)
                # ax.text(0.5, -0.05, 'Some Text', transform=ax.transAxes, verticalalignment='center',
                #         horizontalalignment='center', color='white', fontsize=15)

                ax.axis('off')
                pdf_pages.savefig(fig)
                plt.close(fig)

    def show(self):
        num_pages = ceil(len(self.df) / self.rows_per_page)
        for page in range(num_pages):
            start = page * self.rows_per_page
            end = (page + 1) * self.rows_per_page
            fig, ax = plt.subplots(figsize=(20, 6))
            ax.set_facecolor('#eafff5')
            ax.set_title(f'{self.title} (Page {page+1}/{num_pages})')
            table_data = self.df.iloc[start:end]
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
            self._style_cells(table, ax,table_data)
            self.setup_table(table,table_data)
            self._add_annotations_to_figure(fig)
            ax.axis('off')
            plt.show()

class OrangeTable(Table):
    def setup_table(self, table, table_data):
        if table_data.shape[1] >= table_data.shape[0]:  # Check if columns >= rows
            num_rows = len(table_data) + 1  # account for header row in matplotlib table
            num_cols = len(self.df.columns)
            for i in range(num_cols-1, -1, -1):
                for j in range(num_rows-1, num_cols - i, -1):
                    print(f"{j} {i}")
                    table[j, i].set_facecolor('lightgrey')
        super().setup_table(table, table_data)


class RegularTable(Table):
    pass

class FlexibleColorTable(Table):
    def __init__(self, df, title, color_func, rows_per_page=25):
        super().__init__(df, title, rows_per_page)
        self.color_func = color_func

    def _style_cells(self, table, ax, table_data):
        num_rows, num_cols = len(table_data), len(table_data.columns)
        for i in range(num_rows):
            for j in range(num_cols):
                cell = table[i + 1, j]  # +1 to account for header row
                value = table_data.iat[i, j]
                color = self.color_func(value)
                cell.set_facecolor(color)
                cell.set_text_props(color='white' if self._is_dark(color) else 'black')

    @staticmethod
    def _is_dark(color):
        """Determine if a color is dark based on its RGB values."""
        r, g, b = matplotlib.colors.to_rgb(color)
        return (r * 0.299 + g * 0.587 + b * 0.114) <= 0.5
def color_func(value):
    if pd.api.types.is_numeric_dtype(value):
        return 'red' if value > 80 else 'lightgreen'
    else:
        return 'lightgrey'  # default color for non-numeric cells




# Create a list of 36 unique column names using ascii letters and digits

n_rows = 24
n_cols = 36

# Generate random 6-letter strings
random_strings = [''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=6)) for _ in range(n_rows)]

# Create DataFrame
df = pd.DataFrame()

# Add first 3 columns of 6-letter strings
for i in range(3):
    df[f'col{i+1}'] = random_strings

# Add remaining columns of random integers
for i in range(3, n_cols):
    df[f'col{i+1}'] = np.random.randint(0, 100, size=n_rows)


# Create a table using the base Table class
table1 = Table(df, "Base Table")
table1.add_annotations("This is a base table with random data.")
table1.save("base_table.pdf")

# Create a table using the OrangeTable class
table2 = OrangeTable(df, "Orange Table")
table2.add_annotations(["This is an orange table.", "It has a different style than the base table."])
table2.save("orange_table.pdf")

# Create a table using the RegularTable class
table3 = RegularTable(df, "Regular Table")
table3.add_annotations("This is a regular table.")
table3.save("regular_table.pdf")

table = FlexibleColorTable(df, 'Flexible Color Table', color_func)
table.save("flex_table.pdf")
