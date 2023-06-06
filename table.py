import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import random
import string
import matplotlib.colors as mcolors

# Function to generate a random string of fixed length
def random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# Function to generate a random 4 digit integer
def random_four_digit_int():
    return random.randint(1000, 9999)

# Generate DataFrame with random data
n_rows = 240
df = pd.DataFrame({
    f'column_{i}': [random_string() for _ in range(n_rows)] if i < 3 else [random_four_digit_int() for _ in range(n_rows)] for i in range(40)
})

# Create a PDF object
pdf_pages = PdfPages('output4.pdf')

# Define number of rows per page and number of pages
rows_per_page = 24
pages = len(df) // rows_per_page + 1

# Loop through all pages
for page in range(pages):
    # Extract the slice of data for the current page
    data_slice = df.iloc[page * rows_per_page : (page + 1) * rows_per_page]

    # Check if the data slice is not empty
    if not data_slice.empty:
        # Create a new figure and set the size to fit the full page
        fig, ax = plt.subplots(figsize=(12, 11))  # A4 size

        # Add a title to the figure
        fig.suptitle(f'Data Table - Page {page + 1}', fontsize=24)

        # Adjust the spacing to give more room for the title
        plt.subplots_adjust(top=0.9, bottom=0.2)

        # Disable axis
        ax.axis('tight')
        ax.axis('off')

        # Create a table and add it to the figure
        table = plt.table(
            cellText=data_slice.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )

        # Scale the table to fit the full page
        table.scale(1, 1.5)

        # Set the line width (border thickness) and color
        cell_dict = table.get_celld()
        for key, cell in cell_dict.items():
            if key[0] == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(mcolors.CSS4_COLORS['maroon'])
                cell.set_fontsize(14)  #
                cell.set_width(0.1)# Adjust as necessary to fit your needs
                # Set border thickness to 1.5
                cell.set_edgecolor('black')  # Set border color to yellow
            else:
                if key[1] <= 2:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_width(0.1)
                    cell.set_fontsize(14)  # Adjust as necessary to fit your needs
                    cell.set_facecolor('grey')
                else:
                    cell.set_width(0.1)
                    cell.set_fontsize(12)  # Adjust as necessary to fit your needs
                    cell.set_text_props(multialignment='left')
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor(mcolors.CSS4_COLORS['navy'])

            cell.set_linestyle('-')
            cell.set_linewidth(0.1)  # Adjust as necessary to fit your needs
        table.auto_set_font_size(True)
        # Save the figure to the PDF file

        pdf_pages.savefig(fig, bbox_inches="tight")

# Close the PDF object
pdf_pages.close()
