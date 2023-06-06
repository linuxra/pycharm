import pandas as pd
from pandas.io.formats.style import Styler
from typing import Dict, Any
from IPython.display import display, HTML
import numpy as np


class DataFrameStyler:
    """
    A class for styling pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def highlight_max(self, color: str = 'yellow') -> Styler:
        """
        Highlight the maximum in a Series or DataFrame.
        """
        return self.dataframe.style.highlight_max(color=color)

    def highlight_min(self, color: str = 'lightblue') -> Styler:
        """
        Highlight the minimum in a Series or DataFrame.
        """
        return self.dataframe.style.highlight_min(color=color)

    def background_gradient(self, cmap: str = 'PuBu') -> Styler:
        """
        Color the background in a gradient style.
        """
        return self.dataframe.style.background_gradient(cmap=cmap)

    def bar(self, color: str = 'lightblue', align: str = 'zero') -> Styler:
        """
        Draw bar charts in DataFrame cells.
        """
        return self.dataframe.style.bar(color=color, align=align)

    def highlight_greater_than(self, value: float, color: str = 'yellow') -> Styler:
        """
        Highlight cells greater than a certain value.
        """

        def color_if_greater(val: float) -> str:
            color_str = f'background-color: {color}' if val > value else ''
            return color_str

        return self.dataframe.style.applymap(color_if_greater)

    def set_precision(self, precision: int = 2) -> Styler:
        """
        Set the precision of the numbers in the DataFrame.
        """
        return self.dataframe.style.format("{:." + str(precision) + "f}")

    def apply_custom_css(self, styles: Dict[str, Dict[str, Any]]) -> Styler:
        """
        Apply custom CSS styling.
        """
        css = [{"selector": selector, "props": [(prop, value) for prop, value in props.items()]} for selector, props in
               styles.items()]

        return self.dataframe.style.set_table_styles(css)


np.random.seed(0)  # for reproducibility
df = pd.DataFrame(np.random.rand(30, 10), columns=list('ABCDEFGHIJ'))

# Define the styling rules
styles = {
    'table': {
        'border-collapse': 'collapse',
        'width': '100%'
    },
    'th': {
        'border': '1px solid black',
        'text-align': 'center',
        'background-color': 'lightgray'
    },
    'td': {
        'border': '1px solid black',
        'text-align': 'center'
    }
}

# Apply the styling rules
styler = DataFrameStyler(df)
styled_df = styler.apply_custom_css(styles)
print(styled_df.render())
