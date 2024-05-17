from IPython.display import HTML


def generate_table_of_contents(content):
    # Initialize an empty string to store the HTML code
    toc_html = "<ul>"

    # Iterate over the content and generate HTML code for each item
    for item in content:
        # Add a list item for each item in the content
        toc_html += f'<li><a href="#{item["id"]}">{item["title"]}</a>'
        # Check if the item has subitems and add them as a sublist
        if "subitems" in item:
            toc_html += "<ul>"
            for subitem in item["subitems"]:
                toc_html += (
                    f'<li><a href="#{subitem["id"]}">{subitem["title"]}</a></li>'
                )
            toc_html += "</ul>"
        toc_html += "</li>"

    # Close the unordered list tag
    toc_html += "</ul>"

    return toc_html


def generate_content(content):
    # Initialize an empty string to store the HTML code
    content_html = ""

    # Iterate over the content and generate HTML code for each section
    for item in content:
        # Add a div for each section with an ID
        content_html += f'<div id="{item["id"]}"><h2>{item["title"]}</h2></div>'
        # Check for subitems and add them to the content
        if "subitems" in item:
            for subitem in item["subitems"]:
                content_html += (
                    f'<div id="{subitem["id"]}"><h3>{subitem["title"]}</h3></div>'
                )

    return content_html


# Example content with subitems
content = [
    {
        "id": "section1",
        "title": "Section 1",
        "subitems": [
            {"id": "sub1", "title": "Subsection 1"},
            {"id": "sub2", "title": "Subsection 2"},
        ],
    },
    {"id": "section2", "title": "Section 2"},
    {
        "id": "section3",
        "title": "Section 3",
        "subitems": [{"id": "sub3", "title": "Subsection 3"}],
    },
]

# Generate the table of contents HTML
toc_html = generate_table_of_contents(content)

# Generate the content HTML
content_html = generate_content(content)

# Combine the table of contents and content HTML
full_html = f"""
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
"""

# Display the HTML
HTML(full_html)
