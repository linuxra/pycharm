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
