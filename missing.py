import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# assuming df is your dataframe
df = ...

# Compute percentage of missing data
missing_data = df.isnull().sum() / len(df) * 100

# Create a dataframe for missing data analysis
missing_df = pd.DataFrame({'column_name': df.columns,
                           'percent_missing': missing_data})

# Visualize missing values
msno.matrix(df)
plt.show()

# Generate report
for column in df.columns:
    num_missing = df[column].isnull().sum()
    pct_missing = np.mean(df[column].isnull())
    print(f'{column} - Missing: {num_missing} ({pct_missing: .1%})')

# Visualize the percentage of missing values in a bar chart
missing_df.sort_values('percent_missing', inplace=True)
plt.figure(figsize=(15, 8))
sns.barplot(x='percent_missing', y='column_name', data=missing_df, palette='viridis')
plt.xlabel('Missing Values (%)')
plt.ylabel('Features')
plt.title('Percentage of Missing Values by Feature')
plt.show()
