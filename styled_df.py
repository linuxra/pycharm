


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
