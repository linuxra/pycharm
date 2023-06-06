import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from string import ascii_letters
import seaborn as sns

class Table:
    def __init__(self, df, title):
        self.df = df
        self.fig, self.ax = plt.subplots(figsize=(20, 6))
        self.ax.set_title(title)
        self.table = self.ax.table(cellText=self.df.values, colLabels=self.df.columns, loc='center')

    def setup_table(self):
        self.table.set_fontsize(14)
        self.ax.axis('off')

    def save(self, pdf_pages):
        self.setup_table()
        pdf_pages.savefig(self.fig)

    def show(self):
        self.setup_table()
        plt.show()


class OrangeTable(Table):
    def setup_table(self):
        num_rows = len(self.df) + 1  # account for header row in matplotlib table
        num_cols = len(self.df.columns)
        for i in range(num_cols-1, -1, -1):
            for j in range(num_rows-1, num_cols - i, -1):
                print(f"{j} {i}")
                self.table[j, i].set_facecolor('orange')
        super().setup_table()


class GradientTable(Table):
    def setup_table(self):
        num_rows = len(self.df) + 1  # account for header row in matplotlib table
        num_cols = len(self.df.columns)
        for i in range(num_cols):
            self.table[0, i].set_facecolor('lightgrey')

            # Set color for the first two rows
        for i in range(num_rows):
            self.table[i, 0].set_facecolor('lightblue')  # First row after the header
            self.table[i, 1].set_facecolor('lightblue')  # Second row after the header

        for i in range(num_cols-1, 14, -1):  # Start from 15th column (0-based index)
            for j in range(num_rows-1, num_cols - 1 - i, -1):
                if j > 0 and isinstance(self.df.iat[j-1, i], str) and self.df.iat[j-1, i].isdigit():  # Ensure value is numeric string
                    value = int(self.df.iat[j-1, i])
                    color = sns.light_palette("orange", as_cmap=True)(value/100)  # Generate color based on cell value
                    self.table[j, i].set_facecolor(color)
        super().setup_table()

class RegularTable(Table):
    pass


# Set the number of rows and columns
num_rows = 24
num_cols = 38

# Generate random strings for the first two columns
col1 = [''.join(np.random.choice(list(ascii_letters), size=5)) for _ in range(num_rows)]
col2 = [''.join(np.random.choice(list(ascii_letters), size=5)) for _ in range(num_rows)]

# Generate random integers for the other columns
other_cols = np.random.randint(low=1, high=100, size=(num_rows, num_cols-2))

# Combine the columns into a single DataFrame
df = pd.DataFrame(np.column_stack([col1, col2, other_cols]), columns=[f'col{i}' for i in range(1, num_cols+1)])

# Open the PDF file
with PdfPages('output.pdf') as pdf_pages:

    # Create a regular table and save to PDF
    table1 = RegularTable(df, "Regular Table")
    table1.save(pdf_pages)

    # Create a table with orange cells and save to PDF
    table2 = OrangeTable(df, "Orange Table")
    table2.save(pdf_pages)

    table3 = GradientTable(df, "Gradient Table")
    table3.save(pdf_pages)
