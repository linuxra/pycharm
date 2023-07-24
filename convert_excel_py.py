import pandas as pd
from pprint import pformat
import ast

# Read the Excel file
with pd.ExcelFile('settings5.xlsx') as xls:
    df_paths = pd.read_excel(xls, 'paths')
    df_settings = pd.read_excel(xls, 'settings')

# Convert the paths DataFrame back to a dictionary
paths_dict = df_paths.set_index('Path')['Value'].to_dict()

# Convert settings DataFrame back to a nested dictionary
settings_dict = {}
for row in df_settings.itertuples():
    d = settings_dict
    for key in row[1:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[row[-2]] = row[-1]

# Open settings4.py file in write mode
with open('settings6.py', 'w') as f:
    # Write paths to settings4.py
    for key, value in paths_dict.items():
        if '[' in value and ']' in value:  # Check if value is list
            value = ast.literal_eval(value)  # Convert string to list
        f.write(f"{key} = {pformat(value)}\n")

    # Write settings to settings4.py
    f.write(f"\nsettings = {pformat(settings_dict)}\n")

print("settings4.py file has been written")
