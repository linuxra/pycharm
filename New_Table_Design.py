import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Tuple
from math import ceil


class Table:
    def __init__(self, df: pd.DataFrame, title: str, category: str = None, rows_per_page: int = 24,
                 display_columns: int = 10, repeat_columns: int = 0, scale_x: float = 1, scale_y: float = 1.5,
                 first_columns_to_color: int = 1,
                 header_facecolor: str = '#E61030', first_columns_facecolor: str = '#ADD8E6',
                 other_columns_facecolor: str = '#FFFFFF',
                 fig_size: Tuple[float, float] = (20, 11.7 * 0.9)):
        self.df = df
        self.title = title
        self.category = category
        self.rows_per_page = rows_per_page
        self.display_columns = display_columns
        self.repeat_columns = repeat_columns
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.first_columns_to_color = first_columns_to_color
        self.header_facecolor = header_facecolor
        self.first_columns_facecolor = first_columns_facecolor
        self.other_columns_facecolor = other_columns_facecolor
        self.fig_size = fig_size
        self.annotations = []
        self.df_subset = None

    def _style_cells(self, table, ax, table_data):
        row_colors = ['#f1f1f2', 'w']
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(self.header_facecolor)
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                if col < self.first_columns_to_color:
                    cell.set_facecolor(self.first_columns_facecolor)
                else:
                    cell.set_facecolor(row_colors[row % len(row_colors)])
                cell.set_text_props(color='black', ha='center', alpha=1)
            cell.set_edgecolor('none')

    def add_annotations(self, annotations: List[str]):
        if isinstance(annotations, list):
            self.annotations.extend(annotations)
        else:
            self.annotations.append(annotations)

    def _add_annotations_to_figure(self, fig):
        for i, annotation in enumerate(self.annotations, start=1):
            fig.text(0.05, 0.05 - 0.03 * i, annotation, fontsize=10, transform=plt.gcf().transFigure)

    def setup_table(self, table, ax, table_data):
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(table_data.columns))))

    def save(self, file_name: str):
        if self.category:
            unique_categories = self.df[self.category].unique()
            for category in unique_categories:
                self.df_subset = self.df[self.df[self.category] == category]
                self.save_or_show_helper(file_name, category)
        else:
            self.df_subset = self.df.copy()
            self.save_or_show_helper(file_name)

    def save_or_show_helper(self, file_name: str, category: str = None):
        num_pages_rows = ceil(len(self.df_subset) / self.rows_per_page)
        num_pages_cols = ceil((len(self.df_subset.columns) - self.repeat_columns) / self.display_columns)
        total_pages = num_pages_rows * num_pages_cols
        with PdfPages(file_name) as pdf_pages:
            for page_row in range(num_pages_rows):
                for page_col in range(num_pages_cols):
                    start_row = page_row * self.rows_per_page
                    end_row = (page_row + 1) * self.rows_per_page
                    start_col = self.repeat_columns + page_col * self.display_columns
                    end_col = min(self.repeat_columns + (page_col + 1) * self.display_columns,
                                  len(self.df_subset.columns))

                    fig, ax = plt.subplots(figsize=self.fig_size)
                    ax.set_facecolor('#eafff5')
                    page_number = page_row * num_pages_cols + page_col + 1
                    ax.set_title(f'{self.title} (Page {page_number}/{total_pages}) - Category: {category}')
                    table_data = self.df_subset.iloc[start_row:end_row, np.r_[:self.repeat_columns, start_col:end_col]]
                    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
                    table.scale(*[self.scale_x, self.scale_y])
                    self._style_cells(table, ax, table_data)
                    self.setup_table(table, ax, table_data)

                    self._add_annotations_to_figure(fig)
                    ax.axis('off')
                    pdf_pages.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

    def show(self):
        if self.category:
            unique_categories = self.df[self.category].unique()
            for category in unique_categories:
                self.df_subset = self.df[self.df[self.category] == category]
                self.save_or_show_helper(None, category)
        else:
            self.df_subset = self.df.copy()
            self.save_or_show_helper()


# Create some example dataframes
df_rows = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))
df_cols = pd.DataFrame(np.random.randint(0, 100, size=(24, 36)), columns=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
df_category = pd.DataFrame({
    'Category': ['Fruit', 'Fruit', 'Vegetable', 'Vegetable', 'Fruit', 'Vegetable'],
    'Name': ['Apple', 'Banana', 'Carrot', 'Broccoli', 'Cherry', 'Beans'],
    'Color': ['Red', 'Yellow', 'Orange', 'Green', 'Red', 'Green'],
    'Quantity': [5, 7, 10, 2, 15, 20],
    'Price': [0.5, 0.3, 0.7, 1.2, 0.2, 1.0]
})

# Create table for the example with many rows
table_rows = Table(df_rows, 'Table with Many Rows')
table_rows.save('table_rows.pdf')

# Create table for the example with many columns
table_cols = Table(df_cols, 'Table with Many Columns', display_columns=10, repeat_columns=3)
table_cols.save('table_cols.pdf')

# Create table for the example with category
table_category = Table(df_category, 'Table with Category', category='Category')
table_category.save('table_category.pdf')
