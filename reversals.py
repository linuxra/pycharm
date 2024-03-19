# Here's the entire code for the 'calc_reversals' function with all enhancements and documentation:
# Here's the entire code for the 'calc_reversals' function with all enhancements and documentation:
import pandas as pd
import numpy as np
from scipy.stats import norm


def calc_reversals(df):
    """
    Enhances a DataFrame by calculating various statistics related to reversals. This function:
    - Calculates the variance between expected and actual event rates.
    - Determines the lower 90% CI and upper bound of the variance.
    - Checks if the variance is statistically significant.
    - Calculates the reversal margin at the 90% CI.
    - Computes a one-sided confidence lower bound.
    - Determines if there's a major reversal and row-wise reversal.
    - Adds a 'Total' row with aggregate values and percentages for relevant columns.

    Args:
    df (DataFrame): A pandas DataFrame with columns 'act_bad_rate', 'act_cnts', 'exp_bad_rate', 'exp_cnts'.

    Returns:
    DataFrame: The enhanced DataFrame with additional columns and a 'Total' row.
    """
    # Calculating variance event rate
    df['variance_event_rate'] = df['exp_bad_rate'] - df['act_bad_rate']

    # Standard deviation for confidence intervals
    std_dev = norm.ppf(0.9) * np.sqrt(
        df['act_bad_rate'] * (1 - df['act_bad_rate']) / df['act_cnts'] +
        df['exp_bad_rate'] * (1 - df['exp_bad_rate']) / df['exp_cnts']
    )

    # Lower 90% CI and upper bound of the variance
    df['lower_90_CI'] = df['variance_event_rate'] - std_dev
    df['upper_bound'] = df['variance_event_rate'] + std_dev

    # Statistical significance of the variance
    df['stat_sign_diff'] = np.where((df['lower_90_CI'] < 0) & (df['upper_bound'] > 0), 'No', 'Yes')

    # Reversal margin at 90% CI
    margin_values = [np.nan]  # No calculation for the first row
    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        current_row = df.iloc[i]
        margin = norm.ppf(0.9) * np.sqrt(
            prev_row['act_bad_rate'] * (1 - prev_row['act_bad_rate']) / prev_row['act_cnts'] +
            current_row['act_bad_rate'] * (1 - current_row['act_bad_rate']) / current_row['act_cnts']
        )
        margin_values.append(margin)
    df['reversal_margin_90CI'] = margin_values

    # One-sided confidence lower bound
    one_side_conf_lb_values = [np.nan]  # No calculation for the first row
    for i in range(1, len(df)):
        prev_row_act_bad_rate = df.iloc[i - 1]['act_bad_rate']
        current_margin = df.iloc[i]['reversal_margin_90CI']
        one_side_conf_lb_values.append(current_margin + prev_row_act_bad_rate)
    df['one_side_conf_lb'] = one_side_conf_lb_values

    # Major reversal
    df['major_rev'] = np.where(
        df.index == 0,
        np.nan,
        np.where(df['act_bad_rate'] > df['one_side_conf_lb'], 'Yes', 'No')
    )

    # Row-wise reversal
    df['row_rev'] = np.where(df['act_bad_rate'] > df.shift(1)['act_bad_rate'], 'Yes', 'No')
    df.at[0, 'row_rev'] = np.nan

    # Creating a totals row
    totals = {
        'decile': 'Total',
        'act_cnts': df['act_cnts'].sum(),
        'act_bad_cnts': df['act_bad_cnts'].sum(),
        'act_bad_rate': df['act_bad_cnts'].sum() / df['act_cnts'].sum(),
        'exp_cnts': df['exp_cnts'].sum(),
        'exp_bad_cnts': df['exp_bad_cnts'].sum(),
        'exp_bad_rate': df['exp_bad_cnts'].sum() / df['exp_cnts'].sum(),
        'variance_event_rate': (df['exp_bad_cnts'].sum() / df['exp_cnts'].sum()) -
                               (df['act_bad_cnts'].sum() / df['act_cnts'].sum()),
    }

    # Percentage calculations for 'stat_sign_diff', 'major_rev', and 'row_rev'
    for col in ['stat_sign_diff', 'major_rev', 'row_rev']:
        totals[col] = df[col].value_counts(normalize=True).get('Yes', 0) * 100

    # Append totals row to the DataFrame
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    return df

# Example usage of the function
# Note: 'df_one_per_dec
# Apply formatting to both positive and negative percentages
def format_all_percentage(value):
    """Formats all values as percentages with brackets for negative values."""
    if pd.isna(value):  # Handle NaN values
        return np.nan
    elif value < 0:
        return f"({abs(value) * 100:.1f}%)"
    else:
        return f"{value * 100:.1f}%"

# Assuming df_calc_reversals_totals is the DataFrame you have
# Re-creating the DataFrame with dummy data for the example
df_calc_reversals_totals = pd.DataFrame({
    'act_bad_rate': [0.023, -0.0034, 0.0045, np.nan],
    'exp_bad_rate': [0.033, -0.002, 0.0055, np.nan],
    'variance_event_rate': [0.01, -0.001, 0.006, np.nan],
    'lower_90_CI': [0.02, -0.004, 0.007, np.nan],
    'upper_bound': [0.03, -0.005, 0.008, np.nan],
    'reversal_margin_90CI': [0.025, -0.0035, 0.009, np.nan],
    'one_side_conf_lb': [0.035, -0.006, 0.01, np.nan]
})

rate_columns = ['act_bad_rate', 'exp_bad_rate', 'variance_event_rate',
                'lower_90_CI', 'upper_bound', 'reversal_margin_90CI', 'one_side_conf_lb']

for col in rate_columns:
    df_calc_reversals_totals[col] = df_calc_reversals_totals[col].apply(format_all_percentage)

# Display the formatted DataFrame
df_calc_reversals_totals


def color_yes_red_green_no(val):
    """
    Colors 'Yes' red and others green in a DataFrame for display in a Jupyter Notebook.
    """
    color = 'red' if val == 'Yes' else 'green'
    return f'color: {color}'

# Replace df_calc_reversals_totals with your actual DataFrame containing 'major_rev' and 'row_rev' columns
styled_df = df_calc_reversals_totals.style.applymap(color_yes_red_green_no, subset=['major_rev', 'row_rev'])

# Display the styled DataFrame
styled_df

import os
import pandas as pd

import os
import pandas as pd

import os
import pandas as pd

def process_csv_files(directory, columns, percent_column):
    all_dataframes = []
    periods = [6, 9, 12, 18, 24]  # Define the periods to be considered

    for filename in os.listdir(directory):
        if filename.endswith('.csv') and 'hrsd' not in filename:
            df = pd.read_csv(os.path.join(directory, filename), usecols=columns)

            # Perform the percentage transformation immediately
            if percent_column in df:
                df[percent_column] = ((df[percent_column] * 100).round(2)).astype(str) + '%'

            all_dataframes.append(df)

    # Concatenate all the dataframes to create a long DataFrame
    long_df = pd.concat(all_dataframes)

    # Now we create the wide format DataFrame
    wide_df = pd.DataFrame()

    for period in periods:
        period_df = long_df[long_df['perfw'] == period].copy()
        period_df.set_index('perfyymm', inplace=True)

        # Rename columns with the period as a suffix
        period_df.rename(columns=lambda x: f"{x}_{period}mo", inplace=True)

        # Join with the wide DataFrame on 'perfyymm'
        wide_df = wide_df.join(period_df, on='perfyymm', how='outer')

    # Order columns as per the original data, with percent_column last
    ordered_cols = [col for col in wide_df.columns if col != percent_column] + [percent_column]
    wide_df = wide_df[ordered_cols]

    return wide_df

# Example usage
directory = 'path_to_directory'
columns = ['perfyymm', 'origyymm', 'perfw', 'SD']
percent_column = 'SD'
df = process_csv_files(directory, columns, percent_column)

print(df)

import os
import pandas as pd


def process_csv_files1(directory, columns, percent_column, sort_by, columns_to_drop_prefix):
    """
    Processes all CSV files in a specified directory (excluding files with 'hrsd' in the name) to:
    1. Keep only the specified columns.
    2. Transform one specified column by multiplying by 100 and converting to a percentage string.
    3. Combine the data from all CSV files into a single DataFrame, pivoting on a specified 'perfyymm' value.
    4. Sort the DataFrame by a specified column in descending order.
    5. Drop columns that start with a specified prefix.

    Parameters:
    - directory: The directory where CSV files are located.
    - columns: A list of column names to keep from the CSV files.
    - percent_column: The name of the column to be converted to a percentage string.
    - sort_by: The column name to sort the DataFrame by (in descending order).
    - columns_to_drop_prefix: The prefix of columns to be dropped from the DataFrame.

    Returns:
    - A pandas DataFrame with the processed data.
    """
    # Initialize an empty list to store DataFrames
    all_dataframes = []

    # Define the periods to be considered for pivoting
    periods = [6, 9, 12, 18, 24]

    # Read CSV files from the directory, skipping files with 'hrsd' in the filename
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and 'hrsd' not in filename:
            df = pd.read_csv(os.path.join(directory, filename), usecols=columns)

            # Apply the percentage transformation immediately if the column is present
            if percent_column in df:
                df[percent_column] = ((df[percent_column] * 100).round(2)).astype(str) + '%'

            # Append the DataFrame to the list
            all_dataframes.append(df)

    # Concatenate all the dataframes to create a long DataFrame
    long_df = pd.concat(all_dataframes)

    # Initialize the wide DataFrame
    wide_df = pd.DataFrame()

    # Pivot and merge data for each period
    for period in periods:
        # Filter the DataFrame for the current period
        period_df = long_df[long_df['perfw'] == period]

        # Pivot the period-specific DataFrame
        pivot_df = period_df.pivot(index='perfyymm', columns='perfw', values=['origyymm', percent_column])

        # Flatten the columns MultiIndex, formatting as 'values_period'
        pivot_df.columns = [f"{val}_{perfw}mo" for val, perfw in pivot_df.columns]

        # Merge with the wide DataFrame using 'perfyymm' as the joining key
        wide_df = pd.merge(wide_df, pivot_df.reset_index(), on='perfyymm',
                           how='outer') if not wide_df.empty else pivot_df.reset_index()

    # Sort the DataFrame by the specified column in descending order
    wide_df.sort_values(by=sort_by, ascending=False, inplace=True)

    # Drop columns that start with the specified prefix
    cols_to_drop = [col for col in wide_df.columns if col.startswith(columns_to_drop_prefix)]
    wide_df.drop(columns=cols_to_drop, inplace=True)

    # Return the processed DataFrame
    return wide_df


# Example usage
directory = 'path_to_your_directory'  # Replace with your directory path
columns = ['perfyymm', 'origyymm', 'perfw', 'SD']  # Replace with columns you need
percent_column = 'SD'  # The column to convert to a percentage
sort_by = 'SD'  # The column to sort by in descending order
columns_to_drop_prefix = 'perf'  # The prefix of columns to drop

# Process the CSV files and print the resulting DataFrame
df = process_csv_files(directory, columns, percent_column, sort_by, columns_to_drop_prefix)
print(df)


import os
import pandas as pd

def process_hrsd_files(directory, key_column):
    """
    Reads all CSV files from the specified directory that contain 'hrsd' in the filename,
    combines them into a single DataFrame, and sorts the DataFrame in descending order
    by the specified key column.

    Parameters:
    - directory: The directory where CSV files are located.
    - key_column: The column name by which to sort the DataFrame in descending order.

    Returns:
    - A pandas DataFrame containing the combined data from the 'hrsd' files, sorted in
      descending order by the key column.
    """
    # Initialize an empty list to store DataFrames
    all_dataframes = []

    # Read CSV files from the directory, including only files with 'hrsd' in the filename
    for filename in os.listdir(directory):
        if 'hrsd' in filename and filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename))
            all_dataframes.append(df)

    # Concatenate all the dataframes to create one dataset
    combined_df = pd.concat(all_dataframes)

    # Sort the DataFrame by the specified key column in descending order
    combined_df.sort_values(by=key_column, ascending=False, inplace=True)

    # Return the processed DataFrame
    return combined_df

# Example usage
directory = 'path_to_your_directory'  # Replace with your directory path
key_column = 'your_key_column'        # Replace with the key column to sort by

# Process the 'hrsd' CSV files and print the resulting DataFrame
df = process_hrsd_files(directory, key_column)
print(df)


from IPython.display import display, HTML
from math import lcm
import lorem

# User inputs
Num_ROWS = 4
columns_each_row = (1, 7, 4, 1)

# Calculate the least common multiple of columns_each_row to get max_cols_in_a_row
max_cols_in_a_row = lcm(*columns_each_row)

# Table title and colors
table_title = "Nature's Serenity, Random Thoughts, and More"
title_bg_color = "#779ECB"  # Dark pastel blue
alternate_color1 = "#FEF3C7"
alternate_color2 = "#BDE0FE"

# Function to generate content for each cell
def generate_content(row_index, col_span):
    bg_color = alternate_color1 if row_index % 2 == 1 else alternate_color2
    content = lorem.paragraph()
    if row_index == 0:
        # Title row
        return f"<th colspan='{col_span}' style='border: 1px solid #ccc; padding: 12px; background-color: {title_bg_color}; text-align: center; font-size: 20px; font-weight: bold; color: #2C3E50;'>{table_title}</th>"
    else:
        # Content rows
        return f"<td colspan='{col_span}' style='border: 1px solid #ccc; padding: 12px; text-align: left; vertical-align: top; background-color: {bg_color};'>{content}</td>"

# Generate the table
html_table = "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;'>"
for row in range(Num_ROWS):
    html_table += "<tr>"
    col_spans = max_cols_in_a_row // columns_each_row[row]  # Determine the colspan for each cell
    for _ in range(columns_each_row[row]):
        html_table += generate_content(row, col_spans)
    html_table += "</tr>"
html_table += "</table>"

# Display the HTML table in Jupyter Notebook
display(HTML(html_table))


from IPython.display import display, HTML
from math import lcm
import lorem

# User inputs
Num_ROWS = 4
columns_each_row = (1, 2, 4, 2)

# Table title
table_title = "Nature's Serenity, Random Thoughts, and More"

# Colors for the title and alternating rows
title_bg_color = "#779ECB"  # A dark pastel blue
alternate_color1 = "#FEF3C7"
alternate_color2 = "#BDE0FE"

# Function to generate special content with bullet points and an attachment
def generate_special_content():
    bullet_points = "<ul>" + "".join(f"<li>{lorem.sentence()}</li>" for _ in range(3)) + "</ul>"
    attachment = "<a href='path_to_your_attachment'>Download Attachment</a>"  # Replace with the actual path
    return bullet_points + attachment

# Function to generate content for each cell
def generate_content(row_index, col_span, is_special=False):
    bg_color = alternate_color1 if row_index % 2 == 1 else alternate_color2
    content = generate_special_content() if is_special else lorem.paragraph()
    if row_index == 0:
        return f"<th colspan='{col_span}' style='border: 1px solid #ccc; padding: 12px; background-color: {title_bg_color}; text-align: center; font-size: 20px; font-weight: bold; color: #2C3E50;'>{table_title}</th>"
    else:
        return f"<td colspan='{col_span}' style='border: 1px solid #ccc; padding: 12px; text-align: left; vertical-align: top; background-color: {bg_color};'>{content}</td>"

# Calculate the total number of columns (maximum columns in a row)
max_cols_in_a_row = lcm(*columns_each_row)

# Generate the table
html_table = "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;'>"
for row in range(Num_ROWS):
    html_table += "<tr>"
    col_spans = ceil(max_cols_in_a_row / columns_each_row[row])
    for col in range(columns_each_row[row]):
        # Use special content in the first cell of the second row as an example
        is_special = row == 1 and col == 0
        html_table += generate_content(row, col_spans, is_special)
    html_table += "</tr>"
html_table += "</table>"

# Display the HTML table in Jupyter Notebook
display(HTML(html_table))

from IPython.display import display, HTML
from math import ceil, lcm
import lorem


class HTMLTableBuilder:
    """
    A class to build a customizable HTML table with content from various sources.

    Attributes:
        num_rows (int): The number of rows in the table.
        columns_each_row (tuple): A tuple indicating the number of columns in each row.
        table_title (str): The title of the table.
        styles (dict): A dictionary for custom CSS styles.
    """

    def __init__(self, num_rows, columns_each_row, table_title, styles=None):
        """
        Constructor for HTMLTableBuilder class.

        Parameters:
            num_rows (int): The number of rows in the table.
            columns_each_row (tuple): The number of columns in each row.
            table_title (str): The title of the table.
            styles (dict, optional): Custom styles for the table.
        """
        self.num_rows = num_rows
        self.columns_each_row = columns_each_row
        self.table_title = table_title
        self.styles = styles or {}
        self.title_bg_color = self.styles.get("title_bg_color", "#779ECB")
        self.alternate_color1 = self.styles.get("alternate_color1", "#FEF3C7")
        self.alternate_color2 = self.styles.get("alternate_color2", "#BDE0FE")
        self.max_cols_in_a_row = lcm(*columns_each_row)

    def validate_content_map(self, content_map):
        """
        Validates the content map structure and values.

        Parameters:
            content_map (dict): The content map to validate.

        Raises:
            ValueError: If position (0, 0) is used in content_map.
            TypeError: If content map keys and values are not properly formatted.
        """
        if (0, 0) in content_map:
            raise ValueError("Position (0, 0) is reserved for the table title and cannot be used in content_map.")
        for key, value in content_map.items():
            if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(x, int) for x in key):
                raise TypeError("Content map keys must be tuples of two integers.")
            if not isinstance(value, dict) or "type" not in value or "content" not in value:
                raise TypeError("Each content_map value must be a dictionary with 'type' and 'content' keys.")

    def format_content(self, content_data):
        """
        Formats the content based on its type.

        Parameters:
            content_data (dict): A dictionary containing 'type' and 'content' keys.

        Returns:
            str: HTML formatted content.
        """
        content_type = content_data.get('type')
        content_items = content_data.get('content', [])

        if content_type == 'bulleted':
            return "<ul>" + "".join(f"<li>{item}</li>" for item in content_items) + "</ul>"
        elif content_type == 'numbered':
            return "<ol>" + "".join(f"<li>{item}</li>" for item in content_items) + "</ol>"
        elif content_type == 'paragraph':
            return "".join(f"<p>{item}</p>" for item in content_items)
        elif content_type == 'heading':
            return "".join(f"<h2>{item}</h2>" for item in content_items)
        elif content_type == 'subheading':
            return "".join(f"<h3>{item}</h3>" for item in content_items)
        else:
            return "".join(content_items)  # Default to plain text

    def generate_content(self, row_index, col_span, content_data):
        """
        Generates HTML content for a single cell.

        Parameters:
            row_index (int): The current row index.
            col_span (int): The colspan value for the cell.
            content_data (dict): The content data for the cell.

        Returns:
            str: HTML for the cell.
        """
        bg_color = self.alternate_color1 if row_index % 2 == 1 else self.alternate_color2
        formatted_content = self.format_content(content_data)
        if row_index == 0:
            return f"<th colspan='{col_span}' style='border: 1px solid #ccc; padding: 12px; background-color: {self.title_bg_color}; text-align: center; font-size: 20px; font-weight: bold; color: #2C3E50;'>{self.table_title}</th>"
        else:
            return f"<td colspan='{ col_span}' style='border: 1px solid #ccc; padding: 12px; text-align: left; vertical-align: top; background-color: {bg_color};'>{formatted_content}</td>"

    def build_table(self, content_map):
        """
        Builds the HTML table based on the specified content.

        Parameters:
            content_map (dict): A dictionary where keys are (row, col) tuples and values are content data dictionaries.

        Returns:
            str: Complete HTML table.
        """
        self.validate_content_map(content_map)
        html_table = "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;'>"
        for row in range(self.num_rows):
            html_table += "<tr>"
            col_spans = ceil(self.max_cols_in_a_row / self.columns_each_row[row])
            for col in range(self.columns_each_row[row]):
                content_key = (row, col)
                content_data = content_map.get(content_key, {'content': ['']})  # Empty string for undefined cells
                html_table += self.generate_content(row, col_spans, content_data)
            html_table += "</tr>"
        html_table += "</table>"
        return html_table


import pandas as pd

def excel_to_content_map_v2(excel_file, sheet_name=0):
    """
    Converts an Excel sheet to a content map compatible with HTMLTableBuilder.

    Parameters:
        excel_file (str): Path to the Excel file.
        sheet_name (int/str, optional): Name or index of the sheet in the Excel file.

    Returns:
        dict: Content map for HTMLTableBuilder.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    content_map = {}
    for _, row in df.iterrows():
        position = (int(row['row']), int(row['column']))
        content_type = row['type']
        content = row['content'].split(', ')  # Assuming list items are separated by ', '
        content_map[position] = {'type': content_type, 'content': content}

    return content_map
# Example usage
excel_file = 'path_to_your_excel_file.xlsx'
content_map_from_excel = excel_to_content_map_v2(excel_file)

table_title = "My Table"
styles = {
    "title_bg_color": "#4B8BBE",
    "alternate_color1": "#ECECEC",
    "alternate_color2": "#FFFFFF"
}

# Initialize HTMLTableBuilder
table = HTMLTableBuilder(3, (2, 2, 2), table_title, styles)

# Build the table with content from Excel
html_table = table.build_table(content_map_from_excel)

# Display the table in a Jupyter Notebook
display(HTML(html_table))


class HTMLTableBuilder5:
    """
    A flexible HTML table builder that supports complex table structures including
    rowspan and colspan, customizable styles, and various content types.

    Attributes:
        table_title (str): The title of the table.
        styles (dict): Custom styles for the table.
        rows (list): A list of tuples representing rows and their styles.
        max_span (int): The maximum span for the title, considering colspan of rows.
    """

    def __init__(self, table_title=None, styles=None):
        """
        Initializes the HTMLTableBuilder.

        Args:
            table_title (str, optional): The title of the table. Defaults to None.
            styles (dict, optional): Custom styles for the table. Defaults to None.
        """
        self.table_title = table_title
        self.styles = styles or {}
        self.rows = []
        self.max_span = 0

    def add_row(self, cells, row_style=None):
        """
        Adds a row to the table.

        Args:
            cells (list of dict): A list where each dict represents a cell in the row.
                                  Keys can include 'content', 'colspan', 'rowspan', 'type', 'style'.
            row_style (str, optional): CSS style for the entire row. Defaults to None.
        """
        total_col_span = sum(cell.get('colspan', 1) for cell in cells)
        self.max_span = max(self.max_span, total_col_span)
        self.rows.append((cells, row_style))

    def _generate_cell_html(self, cell):
        """
        Generates HTML for a single cell.

        Args:
            cell (dict): A dictionary representing the cell with possible keys
                         like 'content', 'colspan', 'rowspan', 'type', 'style'.

        Returns:
            str: HTML string for the cell.
        """
        cell_html = "<td"
        if 'colspan' in cell:
            cell_html += f" colspan='{cell['colspan']}'"
        if 'rowspan' in cell:
            cell_html += f" rowspan='{cell['rowspan']}'"
        if 'style' in cell:
            cell_html += f" style='{cell['style']}'"
        cell_html += ">"
        cell_html += self._format_content(cell.get('type', 'text'), cell.get('content', ''))
        cell_html += "</td>"
        return cell_html

    def _format_content(self, content_type, content):
        """
        Formats content based on its type.
         Args:
            content_type (str): The type of the content (e.g., 'text', 'image').
            content (str, dict, or list): The content to be formatted.

        Returns:
            str: Formatted content as an HTML string.
        """
        if isinstance(content, list):
            # If content is a list, format each item in the list
            return ''.join(
                [self._format_content(item.get('type', 'text'), item.get('content', '')) for item in content])
        elif content_type == 'text':
            return content
        elif content_type == 'bulleted':
            list_items = ''.join(f'<li>{item}</li>' for item in content)
            return f'<ul>{list_items}</ul>'

        elif content_type == 'heading':
            return f"<h3 style='text-align: left;'>{content}</h3>"
        elif content_type == 'paragraph':
            return f"<p>{content}</p>"
        elif content_type == 'image':
            return f"<img src='{content['src']}' alt='{content.get('alt', '')}' " \
                   f"width='{content.get('width', '')}' height='{content.get('height', '')}'>"
        return content

    def _generate_row_html(self, cells, row_style):
        """
        Generates HTML for a single row.

        Args:
            cells (list of dict): List of cell definitions for the row.
            row_style (str): CSS style for the entire row.

        Returns:
            str: HTML string for the row.
        """
        row_html = "<tr"
        if row_style:
            row_html += f" style='{row_style}'"
        row_html += ">"
        row_html += ''.join(self._generate_cell_html(cell) for cell in cells)
        row_html += "</tr>"
        return row_html

    def build_table(self):
        """
        Builds the complete HTML table.

        Returns:
            str: The complete HTML table as a string.
        """
        table_html = "<table"
        if 'table_style' in self.styles:
            table_html += f" style='{self.styles['table_style']}'"
        table_html += ">"

        if self.table_title:
            table_html += f"<tr><th colspan='{self.max_span}' style='text-align: center;'>{self.table_title}</th></tr>"

        for cells, row_style in self.rows:
            table_html += self._generate_row_html(cells, row_style)

        table_html += "</table>"
        return table_html
# Example of a cell with comprehensive styling
cell_style = {
    'background-color': '#FFC3A0',
    'color': 'black',
    'text-align': 'left',
    'font-family': 'Arial, sans-serif',
    'font-size': '14px',
    'border': '1px solid #ddd',
    'padding': '8px'
}

# Use in the add_row method
builder.add_row([
    {'content': 'Example content', 'style': '; '.join(f"{k}: {v}" for k, v in cell_style.items())},
    # ... other cells ...
])
from IPython.display import display, HTML
import lorem  # Ensure the lorem package is installed

# Create an instance of the HTMLTableBuilder with a title
builder = HTMLTableBuilder(table_title="Example Table")

# Define background colors for each cell
bg_colors = [
    "#FFDDC1",  # Title color
    "#FFABAB",  # Large cell (Cell 1)
    "#FFC3A0",  # Cell 2 (summary)
    "#FFAFCC",  # Cell 3
    "#B5EAEA",  # Cell 4
    "#B28DFF"   # Cell 5
]

# Define text colors for each cell
text_colors = [
    "black",    # Title text color
    "white",    # Large cell text color
    "black",    # Cell 2 text color
    "darkblue", # Cell 3 text color
    "green",    # Cell 4 text color
    "purple"    # Cell 5 text color
]

# Set the title row style
builder.styles['table_style'] = f'background-color: {bg_colors[0]}; color: {text_colors[0]};'

# Complex content for different cells
cell_2_summary = lorem.sentence()  # One line summary for Cell 2

cell_3_content = [
    {'type': 'heading', 'content': 'Cell 3 Heading'},
    {'type': 'paragraph', 'content': lorem.paragraph()}
]

cell_4_content = [
    {'type': 'heading', 'content': 'Cell 4 Heading'},
    {'type': 'paragraph', 'content': lorem.paragraph()}
]

cell_5_content = [
    {'type': 'heading', 'content': 'Cell 5 Heading'},
    {'type': 'paragraph', 'content': lorem.paragraph()}
]

# Adding the second row
builder.add_row([
    {'content': cell_3_content, 'rowspan': 2, 'style': f'background-color: {bg_colors[1]}; color: {text_colors[1]};'},
    {'content': cell_2_summary, 'style': f'background-color: {bg_colors[2]}; color: {text_colors[2]};'},
    {'content': cell_4_content, 'style': f'background-color: {bg_colors[3]}; color: {text_colors[3]};'}
])

# Adding the third row
builder.add_row([
    {'content': cell_5_content, 'style': f'background-color: {bg_colors[4]}; color: {text_colors[4]};'}
])

# Build the table
html_table = builder.build_table()

# Display the table in a Jupyter Notebook
display(HTML(html_table))


def _generate_cell_html(self, cell):
    """
    Generates HTML for a single cell.

    Args:
        cell (dict): Dictionary representing the cell properties.

    Returns:
        str: HTML string for the cell.
    """
    cell_html = "<td"

    # Apply colspan and rowspan if specified
    if 'colspan' in cell:
        cell_html += f" colspan='{cell['colspan']}'"
    if 'rowspan' in cell:
        cell_html += f" rowspan='{cell['rowspan']}'"

    # Apply cell styles, including border
    cell_style = "border: 1px solid black;"  # Default cell border
    if 'style' in cell:
        cell_style += cell['style']
    cell_html += f" style='{cell_style}'>"

    # Add the cell content
    cell_html += self._format_content(cell.get('type', 'text'), cell.get('content', ''))
    cell_html += "</td>"
    return cell_html
def build_table(self):
    """
    Builds the complete HTML table.

    Returns:
        str: The complete HTML table as a string.
    """
    table_html = "<table"
    # Apply border-collapse to the table and any additional table styles
    table_style = "border-collapse: collapse;"
    if 'table_style' in self.styles:
        table_style += self.styles['table_style']
    table_html += f" style='{table_style}'>"

    # If there is a table title, add it
    if self.table_title:
        table_html += f"<tr><th colspan='{self.max_span}' style='text-align: center; border: 1px solid black;'>{self.table_title}</th></tr>"

    # Generate HTML for each row
    for cells, row_style in self.rows:
        table_html += self._generate_row_html(cells, row_style)

    table_html += "</table>"
    return table_html


class HTMLTableBuilder6:
    """
    A flexible HTML table builder that supports complex table structures including
    rowspan and colspan, customizable styles, various content types, and a CSS stylesheet.
    """

    def __init__(self, table_title=None, styles=None, css_stylesheet=None):
        self.table_title = table_title
        self.styles = styles or {}
        self.css_stylesheet = css_stylesheet or ""
        self.rows = []
        self.max_span = 0

    def add_row(self, cells, row_style=None):
        total_col_span = sum(cell.get('colspan', 1) for cell in cells)
        self.max_span = max(self.max_span, total_col_span)
        self.rows.append((cells, row_style))

    def _generate_cell_html(self, cell):
        cell_html = "<td"
        if 'colspan' in cell:
            cell_html += f" colspan='{cell['colspan']}'"
        if 'rowspan' in cell:
            cell_html += f" rowspan='{cell['rowspan']}'"
        if 'style' in cell:
            cell_html += f" style='{cell['style']}'"
        cell_html += ">"
        cell_html += self._format_content(cell.get('type', 'text'), cell.get('content', ''))
        cell_html += "</td>"
        return cell_html

    def _format_content(self, content_type, content):
        if content_type == 'text':
            return content
        elif content_type == 'bulleted':
            list_items = ''.join(f'<li>{item}</li>' for item in content)
            return f'<ul>{list_items}</ul>'
        elif content_type == 'heading':
            return f"<h3 >{content}</h3>"
        elif content_type == 'paragraph':
            return f"<p>{content}</p>"
        elif content_type == 'list':
            # Correctly handle a list of mixed content types
            return ''.join(self._format_content(item.get('type', 'text'), item.get('content', '')) for item in content)
        return content

    def _generate_row_html(self, cells, row_style):
        row_html = "<tr"
        if row_style:
            row_html += f" style='{row_style}'"
        row_html += ">"
        row_html += ''.join(self._generate_cell_html(cell) for cell in cells)
        row_html += "</tr>"
        return row_html

    def _generate_stylesheet_html(self):
        if self.css_stylesheet:
            return f"<style>{self.css_stylesheet}</style>"
        return ""

    def build_table(self):
        table_html = "<table"
        if 'table_style' in self.styles:
            table_html += f" style='{self.styles['table_style']}'"
        table_html += ">"

        if self.table_title:
            table_html += f"<tr><th colspan='{self.max_span}' style='text-align: center;'>{self.table_title}</th></tr>"

        for cells, row_style in self.rows:
            table_html += self._generate_row_html(cells, row_style)

        table_html += "</table>"
        return self._generate_stylesheet_html() + table_html

# Defining the CSS
css_stylesheet_advanced = """
table {
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
    color: #333;
    text-align: center;
}
th {
    background-color: #6FA1D2;
    color: white;
    text-align: left;
    padding: 10px;
}
td {
    padding: 10px;
    border-bottom: 1px solid #ddd;text-align: left;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
tr:nth-child(odd) {
    background-color: #EFF6FB;
}
h3 {
    color: #B5C0D0;text-align:left;
}
ul {
    padding-left: 10px;
}
"""
css_stylesheet_enhanced = """
table {
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
    color: #333;
    box-shadow: 0px 0px 10px #aaa; /* Added shadow for depth */
    border-radius: 10px; /* Rounded corners */
    overflow: hidden;
}

thead th {
    background-color: #6FA1D2;
    color: white;
    text-align: center; /* Center-aligned header text */
    padding: 10px;
    position: sticky; /* Sticky header */
    top: 0;
    z-index: 10;
    font-weight: bold; /* Bold header text */
}

td {
    padding: 10px;
    border-bottom: 1px solid #ddd;
    font-style: italic; /* Italicized cell text */
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

tr:nth-child(odd) {
    background-color: #EFF6FB;
}

tr:hover {
    background-color: #d1e0e0; /* Hover effect for rows */
}

h3 {
    color: #4CAF50;
}

ul {
    padding-left: 20px;
}
"""

# Creating the table builder instance
table_builder_advanced = HTMLTableBuilder6(table_title="Advanced Example Table", css_stylesheet=css_stylesheet_enhanced)

# Adding rows with complex content
table_builder_advanced.add_row([{'content': 'Feature', 'type': 'text'}, {'content': 'Description', 'type': 'text'}], row_style="font-weight: bold;")

# Row with a heading and a bulleted list
table_builder_advanced.add_row([
    {'content': 'Design', 'type': 'text'},
    {'content': {'type': 'bulleted', 'content': ['Modern', 'User-friendly', 'Responsive']}, 'type': 'bulleted'}
])

# Row with paragraph text
table_builder_advanced.add_row([
    {'content': 'About', 'type': 'text'},
    {'content': 'This table demonstrates a variety of content types including text, headings, lists, and styled paragraphs.', 'type': 'paragraph'}
])

# Row with a heading and styled paragraph
table_builder_advanced.add_row([
    {'content': 'Purpose', 'type': 'text'},
    {'content': 'Enhancing Presentation',
     'type': 'heading'}
])

# Row with complex mixed content
table_builder_advanced.add_row([
    {'content': 'Features', 'type': 'text'},
    {'content': [{'type': 'heading', 'content': 'Advanced Styling'},
                 {'type': 'paragraph', 'content': 'The table uses advanced CSS for professional, aesthetically pleasing design.'}], 'type': 'list'}
])

# Adding rows with complex content as before...
# [Add rows here similar to the previous examples]

# Creating and displaying the table
html_table_advanced = table_builder_advanced.build_table()
HTML(html_table_advanced)


def add_dataframe(self, dataframe):
    """
    Adds rows to the table from a Pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted to HTML table rows.
    """
    # Add header row
    header_cells = [{'content': col, 'type': 'text'} for col in dataframe.columns]
    self.add_row(header_cells, row_style="font-weight: bold;")

    # Add data rows
    for idx, row in dataframe.iterrows():
        row_cells = [{'content': value, 'type': 'text'} for value in row]
        self.add_row(row_cells)