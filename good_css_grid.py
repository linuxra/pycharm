from IPython.display import display, HTML
import pandas as pd
import lorem

# Generate a list of lorem ipsum sentences
lorem_texts = [lorem.sentence() for _ in range(10)]  # Generate 10 lorem ipsum sentences

# Create a DataFrame
df = pd.DataFrame(lorem_texts, columns=["Lorem Ipsum"])

html_table = df[["Lorem Ipsum"]].to_html(index=False, border=0)


html = """
<style>
.grid-container {
  display: grid;
  grid-template-columns: auto auto auto; /* Three columns of equal width */
  grid-template-rows: auto auto auto auto auto; /* Five rows */
  gap: 10px; /* Space between rows and columns */
  background-color: #f4f4f4; /* Light grey background */
  padding: 10px; /* Padding around the grid */
}

.grid-item {
  background-color: #ffffff;
  border: 1px solid rgba(0, 0, 0, 0.8);
  padding: 20px;
  font-size: 16px;
  text-align: center;
  border-radius: 5px; /* Rounded corners */
  transition: transform 0.2s; /* Smooth transform on hover */
}

.grid-item:hover {
  transform: scale(1.05); /* Slightly increase size on hover */
  cursor: pointer;
}

.item1 { grid-row: 1 / 3; background-color: #ffcccb; } /* Light red */
.item2 { background-color: #add8e6; } /* Light blue */
.item3 { background-color: #90ee90; } /* Light green */
.item4 { background-color: #ffb6c1; } /* Light pink */
.item5 { background-color: #ffffe0; } /* Light yellow */
.item6 { background-color: #dda0dd; } /* Plum */
.item7 { background-color: #9acd32; } /* Yellow Green */
.item8 { background-color: #20b2aa; } /* Light Sea Green */
.item9 { background-color: #87cefa; } /* Sky Blue */
.item10 { background-color: #778899; } /* Light Slate Gray */
.item11 { background-color: #f08080; } /* Light Coral */
.item12 { background-color: #66cdaa; } /* Medium Aqua Marine */
.item13 { background-color: #6495ed; } /* Cornflower Blue */
.item14 { background-color: #ff6347; } /* Tomato */
</style>

<div class="grid-container">
  <div class="grid-item item1">Start with one of the following suggested tasks, or ask me anything using the text box below</div>
  <div class="grid-item item2">Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without a garbage </div>
  <div class="grid-item item3">Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.</div>
  <div class="grid-item item4">4</div>
  <div class="grid-item item5">5
  </div>
  <div class="grid-item item6">6</div>
  <div class="grid-item item7">7</div>
  <div class="grid-item item8">8</div>
  <div class="grid-item item9">9</div>
  <div class="grid-item item10">10</div>
  <div class="grid-item item11">11</div>
  <div class="grid-item item12">12</div>
  <div class="grid-item item13">13</div>
  <div class="grid-item item14">14</div>
</div>
"""

display(HTML(html))


import pandas as pd

# Create a dummy DataFrame
data = {
    'Column1': range(1, 9),
    'Column2': range(10, 18),
    'Column3': range(20, 28),
    'Column4': range(30, 38),
    'Column5': range(40, 48),
}
df = pd.DataFrame(data)

# Convert the DataFrame to HTML
html = df.to_html(index=False)

# Insert CSS styles directly into the HTML for demonstration purposes
css = """
<style>
    body {
        font-family: 'Arial', sans-serif; /* Set a clean, readable font */
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr); /* Define 5 columns */
        gap: 10px; /* Space between cells */
        padding: 10px; /* Padding around the grid */
        margin: 20px; /* Margin around the grid container */
    }
    .grid-item {
        background-color: #f8f9fa; /* Very light grey background for grid items */
        border: 1px solid #dee2e6; /* Border for grid items */
        padding: 8px; /* Padding inside grid items */
        text-align: center; /* Center text inside grid items */
        transition: background-color 0.3s; /* Smooth transition for hover effect */
    }
    .grid-item:hover {
        background-color: #e9ecef; /* Change background on hover */
    }
    .title-row {
        grid-column: 1 / -1; /* Span across all columns */
        background-color: #007bff; /* Bootstrap primary blue */
        color: white; /* White text */
        text-align: center;
        padding: 15px;
        font-size: 24px; /* Larger font size for title */
        font-weight: bold; /* Bold font weight for title */
    }
    .header-item {
        background-color: #6c757d; /* Bootstrap secondary gray */
        color: white; /* White text */
        font-weight: bold; /* Bold font weight for headers */
        font-size: 16px; /* Larger font size for headers */
    }
</style>
"""

# Transform the DataFrame HTML to use divs and CSS Grid
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, 'html.parser')
table = soup.find('table')
rows = table.find_all('tr')

# Create a new div that will act as our grid container
grid_container = soup.new_tag('div', **{'class': 'grid-container'})

# Insert a title row
title_div = soup.new_tag('div', **{'class': 'title-row'})
title_div.string = "My DataFrame Title"
grid_container.append(title_div)

# Process header row
for th in rows[0].find_all('th'):
    header_cell = soup.new_tag('div', **{'class': 'grid-item header-item'})
    header_cell.string = th.get_text()
    grid_container.append(header_cell)

# Process each data row
for row in rows[1:]:  # Skip the header row
    for td in row.find_all('td'):
        data_cell = soup.new_tag('div', **{'class': 'grid-item'})
        data_cell.string = td.get_text()
        grid_container.append(data_cell)

# Replace the table with our new grid container
table.replace_with(grid_container)

# Combine everything into a complete HTML document
html_output = f"<html><head>{css}</head><body>{str(soup)}</body></html>"


# Display the HTML in a Jupyter Notebook or write it to an HTML file
display(HTML(html_output))

