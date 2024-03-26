styles = [
    # Table layout
    {'selector': 'table',
     'props': [('table-layout', 'fixed'),
               ('background-color', '#343a40'),  # Dark background color
               ('color', 'white'),  # White text color
               ('width', '100%'),  # Table width
               ('border-collapse', 'collapse')]},  # Collapses the border

    # Headers: bold, centered text, white text color
    {'selector': 'th',
     'props': [('font-weight', 'bold'),
               ('text-align', 'center'),
               ('background-color', '#454d55'),  # Slightly lighter header background
               ('color', 'white'),
               ('white-space', 'normal'),
               ('padding', '8px'),  # Padding for headers
               ('min-width', '100px')]},

    # Alternating row colors
    {'selector': 'tr:nth-child(odd)',
     'props': [('background-color', '#394045')]},  # Adjust color as needed
    {'selector': 'tr:nth-child(even)',
     'props': [('background-color', '#343a40')]},  # Adjust color as needed

    # Cell borders and padding
    {'selector': 'td, th',
     'props': [('border-style', 'solid'),
               ('border-width', '1px'),
               ('border-color', 'black'),
               ('padding', '8px')]},  # Padding for cells

    # Hover color
    {'selector': 'tr:hover',
     'props': [('background-color', '#2c3036')]},  # Adjust hover color as needed

    # Responsive text size
    {'selector': 'td, th',
     'props': [('font-size', '0.9em')]},  # Adjust text size as needed

    # Text alignment in cells
    {'selector': 'td',
     'props': [('text-align', 'left')]},  # Aligns text in cells to left; change as needed
]

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
