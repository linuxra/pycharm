


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


def add_total_row(df):
    df.loc['Total'] = df.select_dtypes('number').sum()

    # Label the total row in the first column
    df.at['Total', df.columns[0]] = 'Total'

    return df


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a DataFrame with 10 rows and specified columns
df = pd.DataFrame({
    'Rank': range(1, 11),  # Rank from 1 to 10
    'Actuals': np.random.randint(100, 1000, 10),  # Integer values for 'Actuals'
    'Bad Rate': np.random.uniform(0, 1, 10),
    'PBad': np.random.uniform(0, 1, 10),
    'Diff': np.random.uniform(-0.5, 0.5, 10),
    'Mad': np.random.uniform(0, 1, 10)
})

# Base CSS
base_css = """
<style type='text/css'>
  :root {
    --column-color: #1D24CA; /* Color for the first n columns */
    --column-font-color: #ffffff;
    --default-border-color: #ddd; /* Border color for table cells */
    --default-padding: 8px; /* Padding for table cells */
    --default-text-align: left; /* Text alignment for table cells */
    --thead-background-color: #201658; /* Background color for the table header */
    --header-font-color: #ffffff; /* White color for table header font */
    --row-odd-background-color: #C7C8CC; /* Background color for odd rows */
    --row-even-background-color: #F2EFE5; /* Background color for even rows */
    --row-hover-background-color: #A3FFD6; /* Background color on row hover */
    --transition-duration: 0.3s; /* Duration for hover transition */
    --transition-easing: ease; /* Easing function for hover transition */
  }
  .centered-table-container {
    display: flex;
    justify-content: center;
  }
  table.grid-table {
    border-collapse: collapse;
    margin: auto;
  }
  .grid-table th, .grid-table td {
    border: 1px solid var(--default-border-color);
    padding: var(--default-padding);
    text-align: var(--default-text-align);
  }
  .grid-table thead th {
    background-color: var(--thead-background-color);
    color: var(--header-font-color);
  }
  .grid-table tbody tr:nth-child(odd) {
    background-color: var(--row-odd-background-color);
  }
  .grid-table tbody tr:nth-child(even) {
    background-color: var(--row-even-background-color);
  }
  .grid-table tbody tr:hover {
    background-color: var(--row-hover-background-color);
    transform: scale(1.02);
    transition: transform var(--transition-duration) var(--transition-easing);
  }
</style>
"""

# User-defined number of columns to color
num_colored_columns = 1

# Additional CSS for dynamically coloring columns
additional_css = "<style type='text/css'>"
for i in range(1, num_colored_columns + 1):
    additional_css += f"""
    .grid-table td:nth-child({i}) {{
        background-color: var(--column-color);
        color: var(--column-font-color);
    }}
    """
additional_css += "</style>"

# Combine the base and additional CSS
full_css = base_css + additional_css

# Process the DataFrame
processed_df = df.pipe(add_total_row)

# Convert DataFrame to HTML without the index and apply CSS
html_data = processed_df.to_html(index=False, classes="grid-table")
styled_html = '<div class="centered-table-container">' + html_data + '</div>' + full_css

# Display the styled table in Jupyter Notebook
display(HTML(styled_html))

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


def add_total_row(df):
    df.loc['Total'] = df.select_dtypes('number').sum()

    # Label the total row in the first column
    df.at['Total', df.columns[0]] = 'Total'

    return df


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a DataFrame with 10 rows and specified columns
df = pd.DataFrame({
    'Rank': range(1, 11),  # Rank from 1 to 10
    'Actuals': np.random.randint(100, 1000, 10),  # Integer values for 'Actuals'
    'Bad Rate': np.random.uniform(0, 1, 10),
    'PBad': np.random.uniform(0, 1, 10),
    'Diff': np.random.uniform(-0.5, 0.5, 10),
    'Mad': np.random.uniform(0, 1, 10)
})

# Base CSS
base_css = """
<style type='text/css'>
  :root {
    --column-color: #1D24CA; /* Color for the first n columns */
    --column-font-color: #ffffff;
    --default-border-color: #ddd; /* Border color for table cells */
    --default-padding: 8px; /* Padding for table cells */
    --default-text-align: left; /* Text alignment for table cells */
    --thead-background-color: #201658; /* Background color for the table header */
    --header-font-color: #ffffff; /* White color for table header font */
    --row-odd-background-color: #C7C8CC; /* Background color for odd rows */
    --row-even-background-color: #F2EFE5; /* Background color for even rows */
    --row-hover-background-color: #A3FFD6; /* Background color on row hover */
    --transition-duration: 0.3s; /* Duration for hover transition */
    --transition-easing: ease; /* Easing function for hover transition */
  }
  .centered-table-container {
    display: flex;
    justify-content: center;
  }
  table.grid-table {
    border-collapse: collapse;
    margin: auto;
  }
  .grid-table th, .grid-table td {
    border: 1px solid var(--default-border-color);
    padding: var(--default-padding);
    text-align: var(--default-text-align);
  }
  .grid-table thead th {
    background-color: var(--thead-background-color);
    color: var(--header-font-color);
  }
  .grid-table tbody tr:nth-child(odd) {
    background-color: var(--row-odd-background-color);
  }
  .grid-table tbody tr:nth-child(even) {
    background-color: var(--row-even-background-color);
  }
  .grid-table tbody tr:hover {
    background-color: var(--row-hover-background-color);
    transform: scale(1.02);
    transition: transform var(--transition-duration) var(--transition-easing);
  }
</style>
"""

# User-defined number of columns to color
num_colored_columns = 1

# Additional CSS for dynamically coloring columns
additional_css = "<style type='text/css'>"
for i in range(1, num_colored_columns + 1):
    additional_css += f"""
    .grid-table td:nth-child({i}) {{
        background-color: var(--column-color);
        color: var(--column-font-color);
    }}
    """
additional_css += "</style>"

# Combine the base and additional CSS
full_css = base_css + additional_css

# Process the DataFrame
processed_df = df.pipe(add_total_row)

# Convert DataFrame to HTML without the index and apply CSS
html_data = processed_df.to_html(index=False, classes="grid-table")
styled_html = '<div class="centered-table-container">' + html_data + '</div>' + full_css

# Display the styled table in Jupyter Notebook
display(HTML(styled_html))



from IPython.display import HTML

def generate_table_of_contents(content):
    # Initialize an empty string to store the HTML code
    toc_html = '<ul>'

    # Iterate over the content and generate HTML code for each item
    for item in content:
        # Add a list item for each item in the content
        toc_html += f'<li><a href="#{item["id"]}">{item["title"]}</a></li>'

    # Close the unordered list tag
    toc_html += '</ul>'

    return toc_html

def generate_content(content):
    # Initialize an empty string to store the HTML code
    content_html = ''

    # Iterate over the content and generate HTML code for each section
    for item in content:
        # Add a div for each section with an ID
        content_html += f'<div id="{item["id"]}"><h2>{item["title"]}</h2></div>'

    return content_html

# Example content
content = [
    {"id": "section1", "title": "Section 1"},
    {"id": "section2", "title": "Section 2"},
    {"id": "section3", "title": "Section 3"},
]

# Generate the table of contents HTML
toc_html = generate_table_of_contents(content)

# Generate the content HTML
content_html = generate_content(content)

# Combine the table of contents and content HTML
full_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Table of Contents</title>
</head>
<body>
    <div id="toc">
        <h1>Table of Contents</h1>
        {toc_html}
    </div>
    <div id="content">
        {content_html}
    </div>
</body>
</html>
'''

# Render the full HTML
display(HTML(full_html))


styles = [
    {'selector': '.centered-table-container', 'props': [('display', 'flex'), ('justify-content', 'center')]},
    {'selector': 'table.grid-table', 'props': [('border-collapse', 'collapse'), ('margin', 'auto')]},
    {'selector': '.grid-table th, .grid-table td', 'props': [('border', '1px solid var(--default-border-color)'), ('padding', 'var(--default-padding)'), ('text-align', 'var(--default-text-align)')]},
    {'selector': '.grid-table thead th', 'props': [('background-color', 'var(--thead-background-color)'), ('color', 'var(--header-font-color)')]},
    {'selector': '.grid-table tbody tr:nth-child(odd)', 'props': [('background-color', 'var(--row-odd-background-color)')]},
    {'selector': '.grid-table tbody tr:nth-child(even)', 'props': [('background-color', 'var(--row-even-background-color)')]},
    {'selector': '.grid-table tbody tr:hover', 'props': [('background-color', 'var(--row-hover-background-color)'), ('transform', 'scale(1.02)'), ('transition', 'transform var(--transition-duration) var(--transition-easing)')]},
]

# User-defined number of columns to color
num_colored_columns = 1

additional_styles = []
for i in range(1, num_colored_columns + 1):
    additional_styles.append({'selector': f'.grid-table td:nth-child({i})', 'props': [('background-color', 'var(--column-color)'), ('color', 'var(--column-font-color)')]})

styles += additional_styles


import base64

# File path
file_path = '/mnt/data/temp_test.xlsx'

# Read the file and encode it
with open(file_path, "rb") as file:
    encoded_string = base64.b64encode(file.read()).decode()

# Preparing the data URL
data_url = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded_string}"

# Limit the display length for convenience
display_url = data_url[:50] + "..." if len(data_url) > 50 else data_url
display_url
<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,[BASE64_STRING]" download="your_file_name.xlsx">Download Excel File</a>


import pandas as pd

# Function to apply color based on condition
def color_value(val):
    if isinstance(val, (int, float)):
        val = val * 100
        color = 'green' if val < 25 else 'yellow'
        return f'color: {color};'
    else:
        return ''

# Function to format text
def text_format(val):
    if isinstance(val, (int, float)):
        val = val * 100
        return f'{val:.2f}%'
    else:
        return val

# Custom function that styles and renders the DataFrame
def style_and_render(df):
    styled = df.style.applymap(color_value)
    styled = styled.format(text_format, na_rep="", escape=False)
    return styled.render(escape=False)

# Sample DataFrame
df = pd.DataFrame({
    'A': [0.1, 0.2, 0.3],
    'B': [0.4, 0.5, 0.6],
    'C': ['x', 'y', 'z']  # Non-numeric column for demonstration
})

# Using pipe to apply the custom function
html_output = df.pipe(style_and_render)

# html_output now contains the HTML representation
# It can be displayed in environments that support HTML, like Jupyter Notebook
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': [0.1, 0.2, 0.3],
    'B': [0.4, 0.5, 0.6],
    'C': ['x', 'y', 'z']  # Non-numeric column for demonstration
})

# Custom function to apply color and percentage formatting to numeric columns
def apply_custom_style(df):
    # Function to color values
    def color_value(val):
        if isinstance(val, (int, float)):
            val = val * 100
            color = 'green' if val < 25 else 'yellow'
            return f'color: {color};'
        return ''

    # Apply color to numeric columns and format them
    styled_df = df.style.applymap(color_value)
    return styled_df.format("{:.2f}%", na_rep="", subset=pd.IndexSlice[:, df.select_dtypes(include=[float, int]).columns])

# Sequence of functions to apply
funcs = [
    apply_custom_style,
    # You can add other functions here for more styling
]

# Applying the functions using pipe and hiding the index
styled_df = df.pipe(lambda x: x.style.pipe(*funcs)).hide_index()

# Now you can display styled_df in a Jupyter Notebook to see the styles
# If you need to render this as HTML, you can use styled_df.render()
# Function to apply custom style
def apply_custom_style(df):
    # Function to color values. This is applied after multiplying by 100
    def color_value(val):
        if isinstance(val, (int, float)):
            val = val * 100  # Multiply by 100
            color = 'green' if val < 25 else 'yellow'  # Apply color based on the multiplied value
            return f'color: {color};'
        return ''

    # Apply the color function to each cell in the DataFrame
    styled_df = df.style.applymap(color_value)

    # Format the numeric values: Multiply by 100 and format as percentages
    # This is where the multiplication and formatting are happening
    formatted_df = styled_df.format("{:.2f}%", na_rep="",
                                    subset=pd.IndexSlice[:, df.select_dtypes(include=[float, int]).columns])

    return formatted_df


def process_text_with_ul(cell):
    # Removing <ul> and </ul> tags
    cell = re.sub(r'</?ul>', '', cell)

    # Finding all <li> items
    items = re.findall(r'<li><strong>(.*?)</strong>(.*?)</li>', cell)

    # Processing each item
    processed_items = []
    for word, text in items:
        # Making the word bold and appending the text
        processed_items.append(f"<b>{word}</b>{text}")

    # Joining all items with a line break
    return '<br>'.join(processed_items)


# Apply the new function to the dataframe
df['Processed Text with UL'] = df['Text'].apply(process_text_with_ul)

# Display the result
HTML(df.to_html(escape=False))


def process_columns(df, columns):
    def process_text_with_ul_convert_str(cell):
        # Convert cell to string
        cell_str = str(cell)

        # Removing <ul> and </ul> tags
        cell_str = re.sub(r'</?ul>', '', cell_str)

        # Finding all <li> items
        items = re.findall(r'<li><strong>(.*?)</strong>(.*?)</li>', cell_str)

        # Processing each item
        processed_items = []
        for word, text in items:
            # Making the word bold and appending the text
            processed_items.append(f"<b>{word}</b>{text}")

        # Joining all items with a line break
        return '<br>'.join(processed_items)

    for column in columns:
        df[column] = df[column].apply(process_text_with_ul_convert_str)

    return df


# Apply the function to specific columns using pipe
processed_df = df.pipe(process_columns, ['Text'])

# Display the result
HTML(processed_df.to_html(escape=False))


def process_columns_with_colon(df, columns):
    def process_text_with_ul_convert_str_and_colon(cell):
        # Convert cell to string
        cell_str = str(cell)

        # Removing <ul> and </ul> tags
        cell_str = re.sub(r'</?ul>', '', cell_str)

        # Finding all <li> items
        items = re.findall(r'<li><strong>(.*?)</strong>(.*?)</li>', cell_str)

        # Check if any items were found
        if not items:
            return cell  # Return original cell content if no pattern match

        # Processing each item
        processed_items = []
        for word, text in items:
            # Making the word bold, adding a colon and space, and appending the text
            processed_items.append(f"<b>{word}</b>: {text.strip()}")

        # Joining all items with a line break
        return '<br>'.join(processed_items)

    for column in columns:
        df[column] = df[column].apply(process_text_with_ul_convert_str_and_colon)

    return df


# Apply the improved function to specific columns using pipe
processed_df_with_colon = df.pipe(process_columns_with_colon, ['Text'])

# Display the result
HTML(processed_df_with_colon.to_html(escape=False))

def process_columns_aligned(df, columns):
    def process_text_aligned(cell):
        # Convert cell to string and split into lines
        lines = str(cell).split('\n')

        # Process each line
        processed_lines = []
        for line in lines:
            # Removing <ul> and </ul> tags
            line = re.sub(r'</?ul>', '', line)

            # Finding all <li> items
            items = re.findall(r'<li><strong>(.*?)</strong>(.*?)</li>', line)

            # Check if any items were found
            if not items:
                processed_lines.append(line.strip())  # Strip and keep original line
                continue

            # Process each <li> item
            for word, text in items:
                # Making the word bold, adding a colon and space, and appending the text
                processed_lines.append(f"<b>{word.strip()}</b>: {text.strip()}")

        # Joining all processed lines with a line break
        return '<br>'.join(processed_lines)


def parse_and_create_function(input_string):
    # Parse the input string to get all ranges
    ranges = input_string.split()

    # Initialize the lists for starts and ends
    starts = []
    ends = []

    # Initialize parts of the function
    function_str = "def fico_ranges(score):\n"

    # Loop through each range description
    for r in ranges:
        start_end, _ = r.split('=')
        start, end = start_end.split('-')

        # Append to lists
        starts.append(start)
        ends.append(end)

        # Add conditional statements to function string
        function_str += f"    if score >= {start} and score <= {end}:\n"
        function_str += f"        return '{start}-{end}'\n"

    # Return the final function string and the lists
    return function_str, starts, ends


# Define the input string
input_string = "300-671='300-671' 672-700='672-700'"

# Generate the function and lists
function_code, starts, ends = parse_and_create_function(input_string)

# Print results
print("Function Code:")
print(function_code)
print("Starts:", starts)
print("Ends:", ends)
