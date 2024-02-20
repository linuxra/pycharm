import pandas as pd
from IPython.display import display, HTML

# Define your DataFrame
data = {'Product': ['Desktop'], 'Price': [800], 'Units Sold': [150]}
df = pd.DataFrame(data)


# Define your styling function
def highlight_table(df):
    return df.style.set_table_styles([

        {'selector': '',
         'props': [('border-collapse', 'collapse'), ('border', '1px solid lightgrey')]},
        {'selector': 'tr:nth-child(even)',
         'props': [('border-bottom', '1px solid #ddd')]},
        {'selector': 'tr:nth-child(odd)',
         'props': [('border-bottom', '2px solid #ddd')]},
        {'selector': 'th', 'props': [('text-align', 'left'), ('padding', '8px')]}
    ])


# Apply the styling function
styled_df = highlight_table(df)

# Render the HTML, explicitly setting index=False
html_output = styled_df.to_html(index=False)

# Display in Jupyter Notebook
display(HTML(html_output))
import pandas as pd
from IPython.display import display, HTML

# Create a DataFrame
df = pd.DataFrame({
    'Product': ['Desktop'],
    'Price': [800],
    'Units Sold': [150]
})

# Define your styling function
def highlight_table(df):
    styles = [
        {'selector': '',
         'props': [('border-collapse', 'collapse'), ('border', '1px solid lightgrey')]},
        {'selector': 'tr:nth-child(even)',
         'props': [('background-color', '#f2f2f2')]},  # assuming you want to style even rows differently
        {'selector': 'th',
         'props': [('text-align', 'left'), ('padding', '8px')]}
    ]
    return df.style.set_table_styles(styles).hide_index()

# Apply the styling function
styled_df = highlight_table(df)

# Center the table by wrapping it in a div with margin set to auto
centered_html = f"""
<div style="overflow-x:auto; margin-left: auto; margin-right: auto;">
    {styled_df.render()}
</div>
"""

# Display in Jupyter Notebook
display(HTML(centered_html))
def highlight_table(df):
    styles = [
        {'selector': '',
         'props': [('border-collapse', 'collapse'), ('border', '1px solid lightgrey')]},
        {'selector': 'tr:nth-child(even)',
         'props': [('background-color', '#f2f2f2')]},
        {'selector': 'tr:nth-child(odd)',
         'props': [('background-color', '#ffffff')]},
        {'selector': 'th, td',
         'props': [('text-align', 'center'), ('padding', '8px')]}, # Center text for both th and td
    ]
    return df.style.set_table_styles(styles).hide_index()