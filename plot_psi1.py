import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to assign color based on value
def assign_color(value):
    if 0 <= value <= 10:
        return 'green'
    elif 10 < value <= 20:
        return 'yellow'
    else:
        return 'red'

# Creating dummy data with proportions instead of percentages
dates = pd.date_range('2022-06', '2023-06', freq='M')  # Dates from June 2022 to May 2023
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']  # Four categories
data = {date.strftime("%b_%Y"): np.random.uniform(0, 0.3, len(categories)) for date in dates}
data['cat'] = categories
df = pd.DataFrame(data)

# Multiplying the proportions by 100 for plotting
for col in df.columns[:-1]:  # Exclude 'cat' column
    df[col] = df[col] * 100  # Convert proportions to percentages

# Transforming the dataframe for plotting
df_melted = df.melt(id_vars='cat', var_name='Date', value_name='Percentage')

# Line colors (distinct from marker colors green, yellow, red)
line_colors = ['blue', 'purple', 'cyan', 'magenta']

# Plotting
plt.figure(figsize=(15, 8))
for idx, category in enumerate(categories):
    cat_data = df_melted[df_melted['cat'] == category]
    dates = cat_data['Date'].tolist()  # Converting to list for proper indexing
    percentages = cat_data['Percentage'].tolist()  # Converting to list for proper indexing

    # Plotting each point with its corresponding marker color
    for i in range(len(dates)):
        marker_color = assign_color(percentages[i])
        plt.scatter(dates[i], percentages[i], color=marker_color, marker='o', s=50)

    # Connecting the points with a line of a unique color
    plt.plot(dates, percentages, color=line_colors[idx], label=category)

plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.title('Trended Line Plot with Distinct Marker and Line Colors')
plt.legend()
plt.show()
