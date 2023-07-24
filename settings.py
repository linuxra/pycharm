import pandas as pd
import numpy as np
from pprint import pformat

# Load the Excel file
xlsx = pd.ExcelFile('settings5.xlsx')

# Create the settings dictionary
settings = {}
for sheet in xlsx.sheet_names:
    df = pd.read_excel(xlsx, sheet)
    df = df.where(pd.notnull(df), None)  # Replace NaNs with None

    # Extract unique report IDs
    report_ids = df['Report_ID'].unique()

    for report_id in report_ids:
        # Filter rows for the current report ID
        report_df = df[df['Report_ID'] == report_id]

        # Create a sub-dictionary for the current report
        report_dict = {}
        for _, row in report_df.iterrows():
            _, key, sub_key, value = row
            if sub_key is not None:
                # If there is a sub-key, add a new sub-dictionary or update existing one
                if key in report_dict:
                    report_dict[key].update({sub_key: value})
                else:
                    report_dict[key] = {sub_key: value}
            else:
                # If there is no sub-key, just add the key-value pair
                report_dict[key] = value

        # Add the report dictionary to the settings dictionary
        settings[report_id] = report_dict

print(settings)


with open('settings6.py', 'w') as f:
    f.write('settings = ' + pformat(settings))