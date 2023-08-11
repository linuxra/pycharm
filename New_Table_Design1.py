import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable, Dict, Union, List, Optional
import pandas as pd

class Table:
    def __init__(self, df: pd.DataFrame, title: str, category: Optional[Union[str, List[str]]] = None,
                 rows_per_page: int = 24, display_columns: Optional[int] = None, repeat_columns: int = 0,
                 scale_x: float = 1, scale_y: float = 1.5, first_columns_to_color: int = 1,
                 header_facecolor: str = '#E61030', first_columns_facecolor: str = '#ADD8E6',
                 other_columns_facecolor: str = '#FFFFFF', fig_size: Tuple[float, float] = (20, 11.7 * 0.9)):

        self.df = df
        self.title = title
        self.category = category
        self.rows_per_page = rows_per_page
        self.display_columns = display_columns if display_columns else df.shape[1]
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

        self.validation_checks = {
            "category": self._check_columns_exist,
            "repeat_columns": self._check_less_than_df_columns,
            "first_columns_to_color": self._check_less_than_df_columns
        }

        self._run_validations()

    def _run_validations(self):
        for variable, validation_func in self.validation_checks.items():
            validation_func(getattr(self, variable))

    def _check_columns_exist(self, columns: Union[str, List[str]]) -> None:
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame columns")

    def _check_less_than_df_columns(self, value: int) -> None:
        if value > self.df.shape[1]:
            raise ValueError(f"Value '{value}' cannot exceed DataFrame columns '{self.df.shape[1]}'")

    # Rest of the class implementation


from pydantic import BaseModel, validator
from typing import Union, List


class TableSettings(BaseModel):
    df: pd.DataFrame
    title: str
    category: Union[str, List[str]] = None
    rows_per_page: int = 24
    display_columns: int = None
    repeat_columns: int = 0
    scale_x: float = 1
    scale_y: float = 1.5
    first_columns_to_color: int = 1
    header_facecolor: str = '#E61030'
    first_columns_facecolor: str = '#ADD8E6'
    other_columns_facecolor: str = '#FFFFFF'
    fig_size: Tuple[float, float] = (20, 11.7 * 0.9)

    @validator('category')
    def category_must_be_in_df_columns(cls, category, values):
        df = values.get('df')
        if df is None:
            raise ValueError("df must be set before category")
        if isinstance(category, list):
            for c in category:
                if c not in df.columns:
                    raise ValueError(f"Category '{c}' not found in DataFrame columns")
        elif category and category not in df.columns:
            raise ValueError(f"Category '{category}' not found in DataFrame columns")
        return category


class Table:
    def __init__(self, settings: TableSettings):
        self.df = settings.df
        self.title = settings.title
        self.category = settings.category
        self.rows_per_page = settings.rows_per_page
        self.display_columns = settings.display_columns or self.df.shape[1]
        self.repeat_columns = settings.repeat_columns
        self.scale_x = settings.scale_x
        self.scale_y = settings.scale_y
        self.first_columns_to_color = settings.first_columns_to_color
        self.header_facecolor = settings.header_facecolor
        self.first_columns_facecolor = settings.first_columns_facecolor
        self.other_columns_facecolor = settings.other_columns_facecolor
        self.fig_size = settings.fig_size
        self.annotations = []
        self.df_subset = None

    # ... rest of your Table methods ...


# Usage:
try:
    settings = TableSettings(df=df, title='MyTitle')
    table = Table(settings)
except ValidationError as e:
    print(e)


class Table:
    def __init__(self, df: pd.DataFrame, title: str, rows_per_page: int = 24, repeat_columns: int = 1,
                 display_columns: int = 10, scale_x: float = 1, scale_y: float = 1.5):
        self.df = df
        self.title = title
        self.rows_per_page = rows_per_page
        self.repeat_columns = repeat_columns
        self.display_columns = display_columns
        self.scale_x = scale_x
        self.scale_y = scale_y

    def _style_cells(self, table, ax):
        row_colors = ['#f1f1f2', 'w']
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor(row_colors[row % len(row_colors)])
            cell.set_edgecolor('w')

    def save_or_show_helper(self, file_name: str = None, category: str = None):
        num_pages_rows = ceil(len(self.df_subset) / self.rows_per_page)
        num_pages_cols = ceil((len(self.df_subset.columns) - self.repeat_columns) / self.display_columns)
        total_pages = num_pages_rows * num_pages_cols

        if file_name:
            with PdfPages(file_name) as pdf_pages:
                for page_row in range(num_pages_rows):
                    for page_col in range(num_pages_cols):
                        self.create_and_save_figure(page_row, page_col, total_pages, category, pdf_pages)
        else:
            for page_row in range(num_pages_rows):
                for page_col in range(num_pages_cols):
                    self.create_and_show_figure(page_row, page_col, total_pages, category)

    def create_and_save_figure(self, page_row, page_col, total_pages, category, pdf_pages):
        fig, ax = self.create_figure(page_row, page_col, total_pages, category)
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def create_and_show_figure(self, page_row, page_col, total_pages, category):
        fig, ax = self.create_figure(page_row, page_col, total_pages, category)
        plt.show()

    def create_figure(self, page_row, page_col, total_pages, category):
        start_row = page_row * self.rows_per_page
        end_row = (page_row + 1) * self.rows_per_page
        start_col = self.repeat_columns + page_col * self.display_columns
        end_col = min(self.repeat_columns + (page_col + 1) * self.display_columns, len(self.df_subset.columns))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        page_number = page_row*num_pages_cols + page_col + 1
        ax.set_title(f'{self.title} (Page {page_number}/{total_pages}) - Category: {category}')

        table_data = self.df_subset.iloc[start_row:end_row, np.r_[:self.repeat_columns, start_col:end_col]]
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
        table.scale(*[self.scale_x, self.scale_y])

        self._style_cells(table, ax)
        return fig, ax

    def save(self, file_name: str, category_column: str = None):
        if category_column:
            for category in self.df[category_column].unique():
                self.df_subset = self.df[self.df[category_column] == category]
                self.save_or_show_helper(file_name=f"{file_name}_{category}.pdf", category=category)
        else:
            self.df_subset = self.df
            self.save_or_show_helper(file_name=file_name)

    def show(self, category_column: str = None):
        if category_column:
            for category in self.df[category_column].unique():
                self.df_subset = self.df[self.df[category_column] == category]
                self.save_or_show_helper(category=category)
        else:
            self.df_subset = self.df
            self.save_or_show_helper()


# Example usage
np.random.seed(0)
df_test = pd.DataFrame(np.random.rand(50, 36), columns=[f'Column {i+1}' for i in range(36)])
df_test['Category'] = np.random.choice(['Cat1', 'Cat2', 'Cat3'], df_test.shape[0])

# Creating table object
table_test = Table(df_test, "Test Table", repeat_columns=3, display_columns=11)

# Displaying table in console with categories
table_test.show(category_column='Category')

# Saving table to file with categories
table_test.save(file_name="test_table", category_column='Category')
