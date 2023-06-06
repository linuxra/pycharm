
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Set the number of rows and columns
num_rows = 24
num_cols = 36

# Create column names as t1, t2, ..., t36
col_names = ['t' + str(i) for i in range(1, num_cols + 1)]

# Generate random integers for the dataframe
data = np.random.randint(low=1, high=100, size=(num_rows, num_cols))

# Create the dataframe
df = pd.DataFrame(data, columns=col_names)

# Add a 'rank' column with value 1
df.insert(0, 'rank', 1)

# Add a 'counter' column with values from 1 to 24
df.insert(1, 'counter', range(1, num_rows + 1))
df.insert(2,'var','range')
df.reset_index()

# Create the matplotlib table from the dataframe
fig, ax = plt.subplots(figsize=(28,12))
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# Set the font size for the table

table.set_fontsize(18)

# Set the row heights for the table
table.scale(1, 2)

# Color the specified cells orange
for i in range(16, 39):
    if i == 16:
        table[num_rows, i].set_facecolor('orange')
    else:
        for j in range(0,i-16+1):
            table[num_rows-j, i].set_facecolor('orange')

# Hide the axis and the axis labels
ax.axis('off')

plt.show()
