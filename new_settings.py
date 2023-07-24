import pandas as pd
import settings4
from pandas.core.common import flatten

# Flatten the settings dictionary and adjust to desired output format
flat_settings = []
for key, value in settings4.settings.items():
    for subkey, subvalue in value.items():
        if isinstance(subvalue, dict):
            for k, v in subvalue.items():
                flat_settings.append([key, subkey, k, v])
        else:
            flat_settings.append([key, subkey, '', subvalue])

# Create a DataFrame from the flattened settings
df_settings = pd.DataFrame(flat_settings, columns=["Report_ID", "Key", "Sub-Key", "Value"])

# Gather all paths and colors from the settings module
paths_and_colors = {name: getattr(settings4, name) for name in dir(settings4)
                    if not name.startswith('_') and 'settings' not in name.lower()}

# Handle paths that are lists by joining them with a comma
paths_and_colors = {k: ', '.join(v) if isinstance(v, list) else v for k, v in paths_and_colors.items()}

# Create a DataFrame from the paths and colors
df_paths_and_colors = pd.DataFrame(list(paths_and_colors.items()), columns=['Path', 'Value'])

# Create a Pandas ExcelWriter object
writer = pd.ExcelWriter('settings5.xlsx', engine='xlsxwriter')

# Write DataFrames to an excel
df_paths_and_colors.to_excel(writer, sheet_name='paths_and_colors', index=False)
df_settings.to_excel(writer, sheet_name='settings', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer._save()
