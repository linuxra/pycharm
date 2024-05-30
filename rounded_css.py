# Define the styles for the DataFrame
styles = [
    {
        "selector": "th",
        "props": [
            ("background-color", "#f2f2f2"),
            ("padding", "10px 20px"),
            ("font-weight", "bold"),  # Make header text bold
            ("border-bottom", "1px solid black"),  # Ensure solid black border
        ],
    },
    {
        "selector": "td",
        "props": [
            ("padding", "10px 20px"),
            ("border-bottom", "1px solid black"),
            ("font-weight", "normal"),  # Make cell text normal weight
        ],
    },
]

# Apply styles to the DataFrame
styled_df = df.style.set_table_styles(styles)

# Custom CSS for the flex container, title, and DataFrame
custom_css = """
<style>
.flex-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 20px;
    padding: 0;
}}

.title-container {{
    width: 100%;
}}

.title-container h1 {{
    margin: 0;
    padding: 10px 20px;
    background-color: #4CAF50;
    color: white;
    border-radius: 10px 10px 0 0; /* Rounded corners for the top only */
    border-bottom: 1px solid black; /* Border to match DataFrame */
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}}

.dataframe-container {{
    width: 100%;
}}

.dataframe-container table {{
    border-collapse: collapse;
    border-radius: 0 0 15px 15px; /* Rounded corners for the bottom only */
    overflow: hidden;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border: 1px solid black;  /* Solid black border for the table */
    border-top: none;  /* Remove the top border to connect with the title */
}}

.dataframe-container th, .dataframe-container td {{
    padding: 10px 20px;
    border-bottom: 1px solid black;  /* Solid black border for the cells */
}}

.dataframe-container th {{
    background-color: #f2f2f2;
    font-weight: bold;  /* Make header text bold */
}}

.dataframe-container td {{
    font-weight: normal;  /* Make cell text normal weight */
}}
</style>
"""

# HTML for the title
title_html = '<div class="title-container"><h1>DataFrame Title</h1></div>'

# Convert styled DataFrame to HTML and wrap it in a container for the custom CSS
html = styled_df.to_html()
html_with_container = f'<div class="dataframe-container">{html}</div>'

# Wrap both title and DataFrame in a flex container
flex_container_html = (
    f'<div class="flex-container">{title_html}{html_with_container}</div>'
)

# Display the flex container with custom CSS
display(HTML(custom_css + flex_container_html))
