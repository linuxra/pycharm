import pandas as pd
from pprint import pformat

# Load the data from the Excel file
xls = pd.ExcelFile('settings5.xlsx')

# Load the 'settings' sheet into a DataFrame
df_settings = pd.read_excel(xls, 'settings')

# Handle nested dictionaries in settings
settings_dict = {}
for _, row in df_settings.iterrows():
    key1, key2, subkey, value = row['Report_ID'], row['Key'], row['Sub-Key'], row['Value']

    if key1 not in settings_dict:
        settings_dict[key1] = {}
    if key2 not in settings_dict[key1]:
        settings_dict[key1][key2] = {}

    if pd.notna(subkey):
        settings_dict[key1][key2][subkey] = value
    else:
        settings_dict[key1][key2] = value

# Load the 'paths_and_colors' sheet into a DataFrame
df_paths_and_colors = pd.read_excel(xls, 'paths_and_colors')

# Convert DataFrame to dictionary
paths_and_colors_dict = df_paths_and_colors.set_index('Path')['Value'].to_dict()

# Convert string values to lists where appropriate
for key, value in paths_and_colors_dict.items():
    if ',' in str(value):
        paths_and_colors_dict[key] = [item.strip() for item in str(value).split(',')]

# Write data to settings4.py
with open('settings7.py', 'w') as f:
    for key, value in paths_and_colors_dict.items():
        f.write(f"{key} = {pformat(value)}\n")
    f.write("settings = " + pformat(settings_dict) + "\n")
