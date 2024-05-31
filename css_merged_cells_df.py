from IPython.display import HTML, display, Javascript
import pandas as pd
from bs4 import BeautifulSoup


def sanitize_html(content):
    """
    Sanitize HTML content using BeautifulSoup to ensure it's safe to display.
    """
    soup = BeautifulSoup(content, "html.parser")
    return str(soup)


def display_css_grid(
    rows, columns, data, title, headers, color_columns=[], merges=None
):
    """
    Display a CSS grid with center-aligned text in merged cells.
    """
    header_rows = len(headers)  # Count of header rows
    merges = merges if merges else []

    # Prepare HTML for grid items
    grid_items = [f'<div class="grid-title">{sanitize_html(title)}</div>']
    current_row = 2  # Start from the second row (after the title)

    # Create headers
    for header_row in headers:
        for header in header_row:
            header_text, start_col, span = header
            grid_items.append(
                f'<div class="grid-header" style="grid-row: {current_row}; grid-column: {start_col} / span {span};">{sanitize_html(header_text)}</div>'
            )
        current_row += 1

    # Handle data cells and merges
    data_start_row = current_row  # Data starts after all headers
    for index, row in data.iterrows():
        r = index + data_start_row
        for c in range(1, columns + 1):
            cell_id = (r, c)
            extra_class = "color-column" if c in color_columns else "grid-item"
            content = sanitize_html(str(row.iloc[c - 1]))
            if any(
                cell_id == (merge[0] + data_start_row - 1, merge[1]) for merge in merges
            ):
                for merge in merges:
                    if cell_id == (merge[0] + data_start_row - 1, merge[1]):
                        r_span, c_span = merge[2], merge[3]
                        grid_items.append(
                            f'<div class="{extra_class}" style="grid-row: {r} / span {r_span}; grid-column: {c} / span {c_span}; display: flex; align-items: center; justify-content: center;">{content}</div>'
                        )
                        break
            elif not any(
                (
                    merge[0] + data_start_row - 1
                    <= r
                    < merge[0] + data_start_row - 1 + merge[2]
                )
                and (merge[1] <= c < merge[1] + merge[3])
                for merge in merges
            ):
                grid_items.append(
                    f'<div class="{extra_class}" style="grid-row: {r}; grid-column: {c};">{content}</div>'
                )

    # Combine all into HTML
    html = f'<div class="grid-container">{"".join(grid_items)}</div>'

    # JavaScript to inject CSS
    css = f"""
    .grid-container {{
        display: grid;
        grid-template-columns: repeat({columns}, 1fr);
        grid-gap: 1px;
        background-color: #e0e0e0;
    }}
    .grid-title {{
        grid-column: 1 / span {columns};
        background-color: #4a86e8;
        color: #ffffff;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
    }}
    .grid-header {{
        background-color: #6fa8dc;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        padding: 10px;
    }}
    .grid-item, .color-column {{
        background-color: #ffffff;
        text-align: center;
        padding: 10px;
        border: 1px solid #cccccc;
    }}
    .color-column {{
        background-color: #b6d7a8;
        font-weight: bold;
    }}
    """
    script = f"""
    var styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = `{css}`;
    document.head.appendChild(styleSheet);
    """
    display(HTML(html))
    display(Javascript(script))


# Example DataFrame and usage as in your provided code.
# Example DataFrame
df = pd.DataFrame(
    {
        "Category": ["A", "B", "C", "D"],
        "PSI": [1, 6, 11, 16],
        "SD": [2, 7, 12, 17],
        "MER": [3, 8, 13, 18],
    }
)

# Title and Headers
title = "Enhanced CSS Grid"
headers = [
    [("QTR", 1, 4)],  # First row of headers spanning all four columns
    [
        ("Category", 1, 1),
        ("PSI", 2, 1),
        ("SD", 3, 1),
        ("MER", 4, 1),
    ],  # Second row of headers each spanning one column
]

# Merges specification (optional)
merges = [(1, 2, 2, 2)]  # Example of merging, optional

# Columns to color differently
color_columns = [1]  # Color the first column differently for emphasis

# Create and display the grid
display_css_grid(4, 4, df, title, headers, color_columns, merges)


def create_headers_from_df(df, header_groups=None):
    """
    Generate a list of header tuples from DataFrame columns, supporting merged headers.

    Args:
        df (pd.DataFrame): The DataFrame from which to derive column names.
        header_groups (list of tuples, optional): Each tuple in the list should contain:
            - a string representing the header title for the merged columns
            - a list of column names that should fall under this header

    Returns:
        list: A list of lists containing header tuples in the format (header_name, start_column, column_span).
    """
    if header_groups is None:
        # No merged headers, create simple one-to-one headers
        header_row = [(col, idx + 1, 1) for idx, col in enumerate(df.columns)]
        return [header_row]

    headers = []
    current_column = 1

    # Process grouped headers for merging
    for header_name, columns in header_groups:
        # Find the starting column index for the first column in this group
        start_col = df.columns.get_loc(columns[0]) + 1
        # Count the number of columns in this group
        span = len(columns)
        headers.append((header_name, start_col, span))
        current_column += span

    # Add another row for individual column headers under each merged header
    individual_headers = []
    for header_name, columns in header_groups:
        for col in columns:
            start_col = df.columns.get_loc(col) + 1
            individual_headers.append((col, start_col, 1))

    return [headers, individual_headers]

# Example Usage:
df = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "PSI": [1, 6, 11, 16],
    "SD": [2, 7, 12, 17],
    "MER": [3, 8, 13, 18]
})

# # Specify header groups for merging
# header_groups = [
#     ("QTR", ["Category", "PSI"]),  # Group 'Category' and 'PSI' under 'QTR'
#     ("Results", ["SD", "MER"])     # Group 'SD' and 'MER' under 'Results'
# ]

headers = create_headers_from_df(df)
print(headers)  # Prints the generated headers with merging specified
