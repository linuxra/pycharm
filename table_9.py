from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import string
from math import ceil
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
import six
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import math

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


class Table:
    def __init__(self, df: pd.DataFrame, title: str, rows_per_page: int = 24, scale_x: float = 1, scale_y: float = 1.5,
                 first_columns_to_color: int = 1,
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

    def render_mpl_table(self, data, col_width=3.0, row_height=0.625, font_size=14,
                         header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0,
                            ax=None, **kwargs):
        size = None
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        print(size)
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
        self.setup_table(mpl_table, ax, data)

        self._style_cells(mpl_table, ax, data)
        ax.set_title('Data Table', pad=20)
        return ax
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






class DataFrameToPDF(Table):
    def __init__(self, df: pd.DataFrame, title: str, plot_title, plot_titles, rows_per_page: int = 24, scale_x: float = 1, scale_y: float = 1.5,
                 first_columns_to_color: int = 1,
                 header_facecolor: str = '#E61030', first_columns_facecolor: str = '#ADD8E6',
                 other_columns_facecolor: str = '#FFFFFF',
                 fig_size: Tuple[float, float] = (20, 11.7 * 0.9)):
        self.df = df
        self.table_title = title
        self.plot_title = plot_title
        self.plot_titles = plot_titles
        super().__init__(df, title, rows_per_page, scale_x, scale_y, first_columns_to_color,
                         header_facecolor, first_columns_facecolor, other_columns_facecolor, fig_size)

    def save_pdf(self, file_name):
        # Create a PdfPages object
        pdf_pages = PdfPages(file_name)

        # Transpose the dataframe for plotting
        df_t = self.df.set_index('Score').T  # exclude the 'data' columns for plotting
        print(df_t)
        # Determine number of rows needed for plots
        num_plots = len(df_t.columns)
        num_rows = math.ceil(num_plots / 3)

        # Create a new figure with enough space for the table and all the plots
        fig = plt.figure(figsize=(20, 2 + 4 * num_rows))  # adjust figure size to accommodate common plot title

        # Add a subplot for the table
        ax_table = plt.subplot2grid((num_rows + 2, 3), (0, 0), colspan=3)
        ax_table.axis('tight')
        ax_table.axis('off')
        table_data = self.df
        table = super().render_mpl_table(table_data, ax=ax_table, header_columns=0, col_width=2.0)
        ax_table.set_title(self.table_title, fontsize=12, weight='bold')

        #if Add common title for all plots
        if num_plots == 10:
            fig.text(0.5, 0.83, self.plot_title, ha='center', fontsize=16)
        else:
            fig.text(0.5, 0.75, self.plot_title, ha='center', fontsize=16)
        # Create subplots for each column
        for idx, column in enumerate(df_t.columns):
            ax_plot = plt.subplot2grid((num_rows + 2, 3), (1 + idx // 3, idx % 3))
            x = np.arange((len(df_t[column].values)))
            ax_plot.plot(x, df_t[column])

            # Set the plot title
            try:
                ax_plot.set_title(f'Plot of {column}')
            except IndexError:
                ax_plot.set_title(f'Plot of {column}')  # default title

            ax_plot.set_xlabel('Index')
            ax_plot.set_ylabel('Value')

        # Adjust the spacing
        plt.tight_layout()

        # Save the figure to the pdf
        pdf_pages.savefig(fig, bbox_inches='tight')

        # Close the pdf
        pdf_pages.close()


np.random.seed(42)  # for reproducible results

# Time variables
time_vars = ['t' + str(i) for i in range(1, 37)]

# Generate unique scores from 1 to 10
scores = np.arange(1, 11)
np.random.shuffle(scores)

# Data
data = np.random.randint(low=1, high=11, size=(10, len(time_vars)))  # scores from 1 to 10

# DataFrame
df1 = pd.DataFrame(data, columns=time_vars)
#
# # Add ID variable with unique scores
df1.insert(0, 'Score', scores[:len(df1)])
np.random.seed(42)
data = np.random.randint(low=int(1e6), high=int(1e9), size=(6, 8))
description = ['desc'+str(i) for i in range(1, 7)]
columns = ['Score', 't2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2022Q2', 't2022Q3', 't2022Q4']
df = pd.DataFrame(data, columns=columns[1:])
df.insert(0, 'Score', description)
# # Create instance of TableWithPlots class
# id =['description']
# Define titles
table_title = "Data Table"
plot_title = "All Plots"
plot_titles = [f'Plot of {col}' for col in df.columns if 't' in col]  # titles for 't' columns

# Create an instance of the class with the DataFrame
pdf_maker = DataFrameToPDF(df, table_title, plot_title, plot_titles)

# Save the DataFrame to a PDF
pdf_maker.save_pdf('output1.pdf')
pdf_maker = DataFrameToPDF(df1, table_title, plot_title, plot_titles)

# Save the DataFrame to a PDF
pdf_maker.save_pdf('output11.pdf')




# np.random.seed(42)
# data = np.random.randint(low=int(1e6), high=int(1e9), size=(6, 8))
# description = ['desc'+str(i) for i in range(1, 7)]
# columns = ['description', 't2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2022Q2', 't2022Q3', 't2022Q4']
# df = pd.DataFrame(data, columns=columns[1:])
# df.insert(0, 'description', description)
# # Create instance of TableWithPlots class
# id =['description']
