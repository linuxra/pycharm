from datetime import datetime, timedelta

def get_date_info(yyyymm, window):
    # Convert yyyymm to datetime object
    current_date = datetime.strptime(yyyymm, "%Y%m")

    # Calculate score_date by subtracting window months
    month = current_date.month - window % 12
    year = current_date.year - window // 12
    if month <= 0:
        month += 12
        year -= 1
    score_date = datetime(year, month, 1)

    # Find start_date and end_date of the score_date month
    start_date = score_date
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    # Format the given yyyymm date as perfyymm (yyMM format)
    perfyymm = current_date.strftime("%y%m")

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), score_date.strftime("%Y%m"), perfyymm

# Example usage
yyyymm = "202012"
window = 10
result = get_date_info(yyyymm, window)
print(result)

from datetime import datetime, timedelta

def generate_monthly_dates(start_date):
    # Parse the start date
    start = datetime.strptime(start_date, "%Y%m")

    # Get the current date
    now = datetime.now()

    # Generate a list to hold the dates
    dates = []

    # Loop from the start date to the current date
    current = start
    while current.year < now.year or (current.year == now.year and current.month < now.month):
        # Append the date in YYYYMM format
        dates.append(current.strftime("%Y%m"))
        # Move to the next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return dates

# Example usage
start_date = "202401"
dates = generate_monthly_dates(start_date)
print(dates)
import pandas as pd
import os
import re
from IPython.display import HTML

def process_csv_files(directory):
    # Initialize an empty dictionary to store the PSI values
    psi_values = {}

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract YYYYMM from the filename
            yyyymm = re.search(r'(\d{6}).csv$', filename)
            if yyyymm:
                yyyymm = yyyymm.group(1)
                # Read the CSV file
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                # Extract the last PSI value and convert to percentage
                last_psi_value = round(df['PSI'].iloc[-1] * 100, 2)
                # Store the value in the dictionary
                psi_values[yyyymm] = last_psi_value

    # Sort the dictionary by keys (YYYYMM) and create a DataFrame
    sorted_keys = sorted(psi_values)
    sorted_psi_values = {k: psi_values[k] for k in sorted_keys}
    df_summary = pd.DataFrame(sorted_psi_values, index=[0])

    # Function to color the PSI values based on conditions
    def color_psi(val):
        color = 'red' if val > 20 else ('yellow' if 10 <= val <= 20 else 'green')
        return f'color: {color}'

    # Apply the coloring function to the DataFrame
    styled_df = df_summary.style.applymap(color_psi)
    return styled_df

# Example usage
directory = 'your_directory_path'
styled_df = process_csv_files(directory)
styled_df

