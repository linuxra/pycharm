import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# assuming df is your dataframe
df = ...

# Compute percentage of missing data
missing_data = df.isnull().sum() / len(df) * 100

# Create a dataframe for missing data analysis
missing_df = pd.DataFrame({'column_name': df.columns,
                           'percent_missing': missing_data})

# Dataframe for messages
messages_df = pd.DataFrame(columns=["Variable", "Message"])

# Check for concept variables (string variables)
concept_vars = df.select_dtypes(include='object').columns
if missing_df.loc[missing_df['column_name'].isin(concept_vars), 'percent_missing'].sum() == 0:
    messages_df = messages_df.append({"Variable": "Concept Variables", "Message": "No missing values."}, ignore_index=True)
else:
    messages_df = messages_df.append({"Variable": "Concept Variables", "Message": "Have missing values and need further auditing."}, ignore_index=True)

# Check for population variables
pop_vars = df.columns[df.columns.str.startswith('pop')]
if missing_df.loc[missing_df['column_name'].isin(pop_vars), 'percent_missing'].sum() == 0:
    messages_df = messages_df.append({"Variable": "Population Variables", "Message": "No missing values."}, ignore_index=True)
else:
    messages_df = messages_df.append({"Variable": "Population Variables", "Message": "Have missing values and need further auditing."}, ignore_index=True)

# Check for month variables
month_vars = df.columns[df.columns.str.startswith(('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'))]
for var in month_vars:
    if missing_df.loc[missing_df['column_name'] == var, 'percent_missing'].values[0] > 10:
        messages_df = messages_df.append({"Variable": var, "Message": f"Has missing percent {missing_df.loc[missing_df['column_name'] == var, 'percent_missing'].values[0]:.1%}"}, ignore_index=True)

# Generate report for other columns (not starting with 'ratio')
other_columns = missing_df[(~missing_df['column_name'].isin(concept_vars)) & (~missing_df['column_name'].isin(pop_vars)) & (~missing_df['column_name'].str.startswith('ratio')) & (missing_df['percent_missing'] > 0)]
for index, row in other_columns.iterrows():
    messages_df = messages_df.append({"Variable": row['column_name'], "Message": f"Missing: {row['percent_missing']: .1%}"}, ignore_index=True)

# Create matplotlib table
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=messages_df.values, colLabels=messages_df.columns, cellLoc = 'center', loc='center')

plt.show()
