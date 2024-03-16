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
