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
