Import necessary libraries
import pandas as pd
from data import your_dictionaries  # Assuming your dictionaries are stored in data.py

# Create a Pandas Excel writer using Openpyxl as the engine
writer = pd.ExcelWriter('output.xlsx', engine='openpyxl')

# Iterate through the dictionaries and their nested dictionaries if any
for dict_name, dictionary in your_dictionaries.items():
    # Convert dictionary to a DataFrame
    df = pd.DataFrame(list(dictionary.items()), columns=['Key', 'Value'])
    
    # Check for nested dictionaries and handle them
    for column in df.columns:
        # If the value is a dictionary, expand it into separate columns
        if df[column].apply(lambda x: isinstance(x, dict)).any():
            nested = df[column].apply(pd.Series)
            df = pd.concat([df.drop(column, axis=1), nested], axis=1)

    # Write DataFrame to a specific sheet
    df.to_excel(writer, sheet_name=dict_name, index=False)

# Save the Excel file
writer.save()