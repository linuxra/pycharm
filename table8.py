import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math


class DataFrameToPDF:
    def __init__(self, df, table_title, plot_title, plot_titles):
        self.df = df
        self.table_title = table_title
        self.plot_title = plot_title
        self.plot_titles = plot_titles

    def save_pdf(self, file_name):
        # Create a PdfPages object
        pdf_pages = PdfPages(file_name)

        # Transpose the dataframe for plotting
        df_t = self.df.drop(['data1', 'data2'], axis=1).T  # exclude the 'data' columns for plotting
        print(df_t)
        # Determine number of rows needed for plots
        num_plots = len(df_t.columns)
        num_rows = math.ceil(num_plots / 3)

        # Create a new figure with enough space for the table and all the plots
        fig = plt.figure(figsize=(10, 2 + 4 * num_rows))  # adjust figure size to accommodate common plot title

        # Add a subplot for the table
        ax_table = plt.subplot2grid((num_rows + 2, 3), (0, 0), colspan=3)
        ax_table.axis('tight')
        ax_table.axis('off')
        ax_table.table(cellText=self.df.values, colLabels=self.df.columns, cellLoc='center')
        ax_table.set_title(self.table_title, fontsize=12, weight='bold')

        # Add common title for all plots
        fig.text(0.5, 0.83, self.plot_title, ha='center', fontsize=16)  # adjust y for proper placement

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


# Create a DataFrame
df = pd.DataFrame(np.random.rand(10, 38), columns=['data1', 'data2'] + [f't{i}' for i in range(1, 37)])

# Define titles
table_title = "Data Table"
plot_title = "All Plots"
plot_titles = [f'Plot of {col}' for col in df.columns if 't' in col]  # titles for 't' columns

# Create an instance of the class with the DataFrame
pdf_maker = DataFrameToPDF(df, table_title, plot_title, plot_titles)

# Save the DataFrame to a PDF
pdf_maker.save_pdf('output.pdf')

