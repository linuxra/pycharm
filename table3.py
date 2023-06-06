import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import string
from math import ceil
import matplotlib.colors as mcolors
import lorem

annotation_text = lorem.paragraph()


class Table:
    def __init__(self, df, title, rows_per_page=24):
        self.df = df
        self.title = title
        self.rows_per_page = rows_per_page
        self.annotations = []

    def _style_cells1(self, table, ax, table_data):
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#003f5c')  # Header color
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                if col < 3:  # The first three columns
                    cell.set_facecolor('#ffa600')  # First three column color
                else:
                    cell.set_facecolor('#bc5090')  # Other cell color
                cell.set_text_props(color='black', ha='center', alpha=1)
            cell.set_edgecolor('none')

    def _style_cells(self, table, ax, table_data):
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4B0082')  # Header color
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                if col < 3:  # The first three columns
                    cell.set_facecolor('#9370DB')  # First three column color
                else:
                    cell.set_facecolor('#F5F5F5')  # Other cell color
                cell.set_text_props(color='black', ha='center', alpha=1)
            cell.set_edgecolor('none')

    def setup_table(self, table, ax, table_data):
        table.auto_set_font_size(False)
        table.set_fontsize(8)  # Reduced fontsize
        #table.scale(1, 3)

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
                fig, ax = plt.subplots(figsize=(20, 11.7 * .9))  # 11.7*0.9 gives 90% of A4 paper height in inches
                ax.set_facecolor('#eafff5')
                ax.set_title(f'{self.title} (Page {page + 1}/{num_pages})')
                table_data = self.df.iloc[start:end]
                table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
                ln = len(table_data)
                result = 1.9 if ln == 24 else 3
                table.scale(1, result)
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.auto_set_column_width(col=list(range(len(self.df.columns))))
                self._style_cells(table, ax, table_data)
                self.setup_table(table, ax, table_data)
                self._add_annotations_to_figure(fig)
                ax.axis('off')
                pdf_pages.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    def show(self):
        num_pages = ceil(len(self.df) / self.rows_per_page)
        for page in range(num_pages):
            start = page * self.rows_per_page
            end = (page + 1) * self.rows_per_page
            fig, ax = plt.subplots(figsize=(20, 11.7 * 0.9))
            ax.set_facecolor('#eafff5')
            ax.set_title(f'{self.title} (Page {page + 1}/{num_pages})')
            table_data = self.df.iloc[start:end]
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
            self._style_cells(table, ax, table_data)
            self.setup_table(table, table_data)
            self._add_annotations_to_figure(fig)
            ax.axis('off')
            plt.show()

class OrangeTable1(Table):
    def __init__(self, df, title, rows_per_page=24):
        super().__init__(df, title, rows_per_page)
        self._header_color = mcolors.CSS4_COLORS['darkorange']
        self._first_three_columns_color = mcolors.CSS4_COLORS['lightsalmon']
        self._other_cell_color = mcolors.CSS4_COLORS['papayawhip']

    @property
    def header_color(self):
        return self._header_color

    @header_color.setter
    def header_color(self, color):
        self._header_color = color

    @property
    def first_three_columns_color(self):
        return self._first_three_columns_color

    @first_three_columns_color.setter
    def first_three_columns_color(self, color):
        self._first_three_columns_color = color

    @property
    def other_cell_color(self):
        return self._other_cell_color

    @other_cell_color.setter
    def other_cell_color(self, color):
        self._other_cell_color = color

    def _style_cells(self, table, ax,table_data):
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(self.header_color)
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                if col < 3:  # The first three columns
                    cell.set_facecolor(self.first_three_columns_color)
                else:
                    cell.set_facecolor(self.other_cell_color)
                cell.set_text_props(color='black', ha='center', alpha=1)
            cell.set_edgecolor('none')

    def setup_table(self, table, ax, table_data):
        table.auto_set_font_size(False)
        table.set_fontsize(14)  # You can adjust this value as needed

        #table.scale(1, 1.0)  # You can adjust these values as needed
        table.auto_set_column_width(col=list(range(len(self.df.columns))))

        if table_data.shape[1] >= table_data.shape[0]:  # Check if columns >= rows
            num_rows = len(table_data) + 1  # account for header row in matplotlib table
            num_cols = len(self.df.columns)
            for i in range(num_cols - 1, -1, -1):
                for j in range(num_rows - 1, num_cols - i, -1):
                    table[j, i].set_facecolor('gainsboro')
class OrangeTable(Table):
    def _style_cells1(self, table, ax, table_data):
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#FF8C00')  # Header color
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                if col < 3:  # The first three columns
                    cell.set_facecolor('#FFA07A')  # First three column color
                else:
                    cell.set_facecolor('#FFEFD5')  # Other cell color
                cell.set_text_props(color='black', ha='center', alpha=1)
            cell.set_edgecolor('none')

    def setup_table(self, table, ax, table_data):
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # You can adjust this value as needed
        #table.auto_set_column_width(col=list(range(len(self.df.columns))))
        #table.scale(1.5, 4)  # You can adjust these values as needed

        if table_data.shape[1] >= table_data.shape[0]:  # Check if columns >= rows
            num_rows = len(table_data) + 1  # account for header row in matplotlib table
            num_cols = len(self.df.columns)
            for i in range(num_cols - 1, -1, -1):
                for j in range(num_rows - 1, num_cols - i, -1):
                    table[j, i].set_facecolor('lightblue')


    def _style_cells(self, table, ax,table_data):
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(mcolors.CSS4_COLORS['darkorange'])  # Header color
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                if col < 3:  # The first three columns
                    cell.set_facecolor(mcolors.CSS4_COLORS['lightsalmon'])  # First three column color
                else:
                    cell.set_facecolor(mcolors.CSS4_COLORS['papayawhip'])  # Other cell color
                cell.set_text_props(color='black', ha='center', alpha=1)
            cell.set_edgecolor('none')


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
table.save("table3.pdf")

table1 = OrangeTable(df, "Orange Table")
table1.add_annotations(["This is an orange table.", "ANOTHER LINE FOR TESTING"])
table1.save("orange_table3.pdf")
table2 = OrangeTable1(df, 'Orange Table')

# Change colors using setters
table2.header_color = mcolors.CSS4_COLORS['orangered']
table2.first_three_columns_color = mcolors.CSS4_COLORS['navajowhite']
table2.other_cell_color = mcolors.CSS4_COLORS['seashell']
table2.add_annotations(annotation_text)
# Save table to a PDF
table2.save("orange_table_setters.pdf")