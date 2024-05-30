from IPython.display import HTML, display
import pandas as pd
from bs4 import BeautifulSoup


def sanitize_html(content):
    """
    Sanitize HTML content using BeautifulSoup to ensure it's safe to display.

    Args:
    content (str): HTML content to sanitize.

    Returns:
    str: Sanitized HTML content.
    """
    soup = BeautifulSoup(content, "html.parser")
    return str(soup)


def create_css_grid(rows, columns, merges, data, title, column_names):
    """
    Create and display a CSS grid in a Jupyter notebook with specified data, title, and column names.

    Args:
    rows (int): Number of rows in the grid.
    columns (int): Number of columns in the grid.
    merges (list of tuples): Each tuple specifies (start_row, start_col, row_span, col_span) for merged cells.
    data (dict): Data to display in the grid, indexed by (row, column).
    title (str): Title of the grid.
    column_names (list): List of column names.
    """
    # CSS setup for the grid container
    style = f"""
    <style>
        .grid-container {{
            display: grid;
            grid-template-columns: repeat({columns}, 1fr);
            grid-template-rows: auto repeat({rows + 2}, 100px);
            gap: 1px;
            background-color: #f0f0f0;  /* Light grey background */
        }}
        .grid-item, .grid-title, .grid-column-name {{
            background-color: #ffffff;  /* White background for cells */
            padding: 10px;
            text-align: center;
            border: 1px solid #aaaaaa;  /* Grey borders */
        }}
        .grid-title {{
            grid-column: 1 / span {columns};
            font-size: 20px;
            font-weight: bold;
            color: #333333;  /* Dark grey color for title text */
            background-color: #cccccc;  /* Grey background for title */
        }}
        .grid-column-name {{
            font-weight: bold;
            color: #ffffff;  /* White text for column names */
            background-color: #666666;  /* Dark grey background for column names */
        }}
    </style>
    """

    # Prepare HTML for grid items
    grid_items = [f'<div class="grid-title">{sanitize_html(title)}</div>']  # Title row
    grid_items += [
        f'<div class="grid-column-name" style="grid-row: 2; grid-column: {c};">{sanitize_html(column_names[c-1])}</div>'
        for c in range(1, columns + 1)
    ]  # Column names row

    # Handle regular cells and merges
    cell_content = {}
    for r in range(3, rows + 3):  # Adjust row start due to title and column names
        for c in range(1, columns + 1):
            cell_content[(r, c)] = data.get(
                (r - 2, c), ""
            )  # Shift data keys to match grid positions

    # Update the dictionary to handle merges
    for start_row, start_col, row_span, col_span in merges:
        for r in range(
            start_row + 2, start_row + row_span + 2
        ):  # Adjust for title and column names
            for c in range(start_col, start_col + col_span):
                if (r, c) != (start_row + 2, start_col):
                    cell_content.pop((r, c), None)
        # Set the main cell of the merge with HTML handling
        merge_content = sanitize_html(data.get((start_row, start_col), "Merged"))
        cell_content[(start_row + 2, start_col)] = (
            f'<div class="grid-item" style="grid-row: {start_row + 2} / span {row_span}; grid-column: {start_col} / span {col_span};">{merge_content}</div>'
        )

    # Add individual cell contents
    for r in range(3, rows + 3):
        for c in range(1, columns + 1):
            if (r, c) in cell_content:
                if isinstance(cell_content[(r, c)], str) and cell_content[
                    (r, c)
                ].startswith("<div"):
                    grid_items.append(
                        cell_content[(r, c)]
                    )  # Merged cell with custom div
                else:
                    grid_items.append(
                        f'<div class="grid-item" style="grid-row: {r}; grid-column: {c};">{sanitize_html(cell_content[(r, c)])}</div>'
                    )

    # Combine all into HTML
    html = f"<div class='grid-container'>{style}{''.join(grid_items)}</div>"

    # Display the grid in the Jupyter notebook
    display(HTML(html))


def dataframe_to_dict(dataframe, rows, columns, merges):
    """
    Convert a DataFrame into a dictionary suitable for use in a CSS grid.

    Args:
    dataframe (pd.DataFrame): Source DataFrame.
    rows (int): Number of rows in the grid.
    columns (int): Number of columns in the grid.
    merges (list of tuples): Merging specifications for cells.

    Returns:
    dict: Dictionary with grid data, keyed by (row, column).
    """
    # Initialize dictionary to hold the grid data
    data_dict = {}

    # Populate the dictionary with DataFrame data
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            data_dict[(i, j)] = str(
                dataframe.iloc[
                    (i - 1) % dataframe.shape[0], (j - 1) % dataframe.shape[1]
                ]
            )

    # Handle merges by ensuring only the top-left cell of each merge area keeps its data
    for start_row, start_col, row_span, col_span in merges:
        merge_value = data_dict[(start_row, start_col)]
        for r in range(start_row, start_row + row_span):
            for c in range(start_col, start_col + col_span):
                data_dict[(r, c)] = merge_value

    return data_dict


# Example DataFrame
df = pd.DataFrame(
    {
        "A": [1, 6, 11, 16],  # Data for column A
        "B": [2, 7, 12, 17],  # Data for column B
        "C": [3, 8, 13, 18],  # Data for column C
        "D": [4, 9, 14, 19],  # Data for column D
        "E": [5, 10, 15, 20],  # Data for column E
    }
)

# Convert DataFrame to dictionary for the grid
grid_data = dataframe_to_dict(df, 4, 5, [(2, 2, 2, 1)])

# Title and Column Names
title = "Sample CSS Grid"
column_names = ["A", "B", "C", "D", "E"]

# Create and display the grid
create_css_grid(4, 5, [(2, 2, 2, 1)], grid_data, title, column_names)

%%javascript
var styleSheets = document.styleSheets;
for (var i = 0; i < styleSheets.length; i++) {
    styleSheets[i].disabled = true;  // Disable all stylesheets
}

