from IPython.display import Markdown

def display_markdown_with_html(section_id, title, content):
    """
    Display formatted Markdown content with embedded HTML in a Jupyter notebook.

    Args:
    section_id (str): HTML anchor id for linking within or across documents.
    title (str): The title for the markdown section.
    content (str): The main content in HTML format.

    Returns:
    None: Renders Markdown formatted output in the notebook.
    """
    markdown_text = f"""
<a id='{section_id}'></a>
## {title}
<div class="custom-Font">
{content}
</div>
"""
    display(Markdown(markdown_text))

# Example usage

display_markdown_with_html('section1', '1. Overview', overview_content)
