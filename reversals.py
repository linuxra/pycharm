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

def process_csv_files(directory, columns, key_columns, percent_column):
    all_dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and 'hrsd' not in filename:
            df = pd.read_csv(os.path.join(directory, filename), usecols=columns)

            # Apply percentage conversion if the column exists in this DataFrame
            if percent_column in df:
                df[percent_column] = (df[percent_column] * 100).astype(str) + '%'

            all_dataframes.append(df)

    # Concatenate to form a long DataFrame
    long_df = pd.concat(all_dataframes)

    # Ensure key_columns is a list or a tuple
    if not isinstance(key_columns, (list, tuple)):
        key_columns = [key_columns]

    # Set multi-index if there are multiple key columns
    if len(key_columns) > 1:
        long_df.set_index(key_columns, inplace=True)

    # Pivot to create a wide table
    wide_df = long_df.pivot_table(index=key_columns[0], columns=key_columns[1:], aggfunc='first')

    return wide_df

# Example usage
directory = 'path_to_directory'
columns = ['column1', 'column2', 'column3']
key_columns = ['key_column1', 'key_column2']
percent_column = 'column_to_percent'
df = process_csv_files(directory, columns, key_columns, percent_column)




