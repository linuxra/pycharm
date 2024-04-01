


def color_conditionally(styler, cols):
    def colorize(val):
        # Convert string percentages to float if necessary
        if isinstance(val, str) and '%' in val:
            val = float(val.strip('%'))

        # Multiple conditions with different colors
        if val > 75:
            color = '#ff9999'  # Red for values greater than 75
        elif val > 50:
            color = '#add8e6'  # Light blue for values greater than 50
        else:
            color = ''
        return f'background-color: {color}'

    return styler.applymap(colorize, subset=cols)
def color_columns(styler, cols, color):
    return styler.applymap(lambda x: f'background-color: {color}', subset=cols)
import pandas as pd

# Example DataFrame
data = {'Column1': [60, '40%', 30], 'Column2': [20, '80%', '100%'], 'Column3': [50, 60, 70]}
df = pd.DataFrame(data)

# Initial styles list (if any)
styles = [{'selector': 'th', 'props': [('font-size', '12pt')]}]

# Columns to style
columns_to_color = ['Column1', 'Column2']
columns_for_conditional = ['Column2', 'Column3']

# Apply styles
styled_df = (df.style
               .pipe(color_columns, columns_to_color, '#add8e6')  # Light blue background for specific columns
               .pipe(color_conditionally, columns_for_conditional)  # Conditional coloring on other columns
               .set_table_styles(styles)
               .hide_index())

# Display styled DataFrame in Jupyter Notebook
styled_df


html = f"""
<div class="flex-container">
    <div class="flex-item">{styled_df1_html}</div>
    <div class="flex-item">{styled_df2_html}</div>
</div>
"""
<!DOCTYPE html>
<html>
<head>
    <title>Flexbox Layout Test</title>
    <style>
        /* Flex container */
        .flex-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            border: 2px solid black; /* Border for the container */
            padding: 10px;
        }

        /* First flex item - full width */
        .flex-item-1 {
            width: 100%;
            background-color: lightblue; /* Background color for the first item */
            border: 2px solid blue; /* Border for the first item */
            padding: 10px;
            box-sizing: border-box; /* Include padding and border in the element's total width and height */
        }

        /* Second flex item - aligned to the right bottom */
        .flex-item-2 {
            align-self: flex-end;
            margin-top: auto; /* This pushes the item to the bottom */
            background-color: lightgreen; /* Background color for the second item */
            border: 2px solid green; /* Border for the second item */
            padding: 10px;
            box-sizing: border-box; /* Include padding and border in the element's total width and height */
        }
    </style>
</head>
<body>
    <div class="flex-container">
        <div class="flex-item-1">Item 1</div>
        <div class="flex-item-2">Item 2</div>
    </div>
</body>
</html>

styles = [
    # Your existing styles...

    # Rule to remove shadow from the table
    {'selector': 'table, tr, th, td',
     'props': [('box-shadow', 'none')]},  # This line removes shadows
]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_data(ax, df, x_col, y_cols, y_labels, title, x_label, y_label):
    for y_col, label in zip(y_cols, y_labels):
        ax.plot(df[x_col], df[y_col], label=label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

# Sample DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'Decile': range(1, 11),
    'Bad': np.random.rand(10),
    'PBad': np.random.rand(10)
})

# Calculating LogOdds and PLogOdds
df['LogOdds'] = np.log(df['Bad'] / (1 - df['Bad']))
df['PLogOdds'] = np.log(df['PBad'] / (1 - df['PBad']))

# Create a 3x2 subplot structure
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# Apply the plotting function to each subplot
for i in range(3):
    plot_data(axs[i, 0], df, 'Decile', ['Bad', 'PBad'], ['Bad', 'PBad'], f'Row {i+1} - Bad and PBad', 'Decile', 'Probability')
    plot_data(axs[i, 1], df, 'Decile', ['LogOdds', 'PLogOdds'], ['LogOdds', 'PLogOdds'], f'Row {i+1} - LogOdds and PLogOdds', 'Decile', 'Log Odds')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()



.flex-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    border: 2px solid black; /* Border for the container */
    padding: 10px;
}

/* First flex item for the small DataFrame */
.flex-item-1 {
    width: 100%;
    background-color: lightblue; /* Background color for the first item */
    border: 2px solid blue; /* Border for the first item */
    padding: 10px;
    box-sizing: border-box; /* Include padding and border in the element's total width and height */
}

/* Second flex item for the title */
.flex-item-2 {
    width: 100%;
    background-color: lightgreen; /* Background color for the title */
    border: 2px solid green; /* Border for the title */
    padding: 10px;
    box-sizing: border-box; /* Include padding and border in the element's total width and height */
    text-align: center; /* Center align the title text */
}

/* Third flex item for the big DataFrame */
.flex-item-3 {
    width: 100%;
    background-color: lightyellow; /* Background color for the big DataFrame */
    border: 2px solid orange; /* Border for the big DataFrame */
    padding: 10px;
    box-sizing: border-box; /* Include padding and border in the element's total width and height */
}
html_string = """
<div class="flex-container">
    <div class="flex-item-1">
        <!-- Small DataFrame HTML here -->
    </div>
    <div class="flex-item-2">
        Title Text
    </div>
    <div class="flex-item-3">
        <!-- Big DataFrame HTML here -->
    </div>
</div>
"""
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

def highlight_last_value(s, color):
    """
    Apply a background color to the last value of a Series.
    """
    return ['background-color: {}'.format(color) if i == len(s) - 1 else '' for i in range(len(s))]

# Apply the styling
styled_df = df.style.apply(highlight_last_value, color='yellow', subset=['B'])
styled_df


def highlight_last_value(styler, column, color):
    """
    Apply a background color to the last value of a specified column.
    """
    def apply_style(s):
        styles = [''] * (len(s) - 1) + [f'background-color: {color}']
        return styles

    return styler.apply(apply_style, subset=[column])



import seaborn as sns

# Set the seaborn grid style
sns.set(style="whitegrid")

# Plotting the data using seaborn with the corrected x-axis
plt.figure(figsize=(10, 6))

sns.lineplot(x=df['perf_vint'], y=df['6mo_sd'], label='6 months')
sns.lineplot(x=df['perf_vint'], y=df['9mo_sd'], label='9 months')
sns.lineplot(x=df['perf_vint'], y=df['12mo_sd'], label='12 months')
sns.lineplot(x=df['perf_vint'], y=df['18mo_sd'], label='18 months')
sns.lineplot(x=df['perf_vint'], y=df['24mo_sd'], label='24 months')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()

plt.xlabel('Performance Vintage (YYYYMM)')
plt.ylabel('Standard Deviation (%)')
plt.title('Trend of Standard Deviation Over Different Time Periods')
plt.legend()
plt.grid(True)

plt.show()

