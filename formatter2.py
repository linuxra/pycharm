import pandas as pd


class DataFrameFormatter:
    def __init__(self, dataframe, format_rules=None, metric_columns=None):
        """
        Initialize the DataFrameFormatter.

        Args:
        - dataframe (pd.DataFrame): The DataFrame to format.
        - format_rules (dict): Formatting rules based on metric columns.
        - metric_columns (list/str): Columns considered as metrics.
        """
        self.original_df = dataframe
        self.dataframe = dataframe.copy()
        self.format_rules = format_rules or {}

        if metric_columns:
            if isinstance(metric_columns, str):
                self.metric_columns = [metric_columns]
            else:
                self.metric_columns = metric_columns

    def _convert_to_number(self, value):
        """Converts a string to a number if possible."""
        try:
            value = value.replace(",", "")
            if "." in value:
                return float(value)
            else:
                return int(value)
        except (ValueError, AttributeError):
            return value

    def apply_default_format(self, inplace=True):
        """Apply the default comma formatting."""
        self.dataframe = self.dataframe.applymap(self._convert_to_number).applymap(self.format_comma)
        if inplace:
            self.original_df.update(self.dataframe)
        else:
            return self.dataframe

    def apply_format(self, format_type, inplace=True):
        """Apply a specified format to the DataFrame."""
        format_functions = {
            "comma": self.format_comma,
            "million": self.format_million,
            "percentage": self.format_percentage
        }
        formatter = format_functions.get(format_type)
        if not formatter:
            raise ValueError(f"Unknown format_type: {format_type}")

        self.dataframe = self.dataframe.applymap(self._convert_to_number).applymap(formatter)

        if inplace:
            self.original_df.update(self.dataframe)
        else:
            return self.dataframe

    def format_based_on_metric(self, inplace=True):
        """Format the DataFrame based on metric columns."""
        if not self.metric_columns:
            raise ValueError("metric_columns must be provided for this method.")

        def format_row(row):
            metric_keys = tuple(row[col] for col in self.metric_columns)
            formatter = self.format_rules.get(metric_keys)
            if formatter:
                for col, value in row.items():
                    if col not in self.metric_columns:
                        try:
                            row[col] = formatter(self._convert_to_number(value))
                        except Exception as e:
                            print(f"Error formatting {value} with {formatter.__name__}: {e}")
            return row

        self.dataframe = self.dataframe.apply(format_row, axis=1)

        if inplace:
            self.original_df.update(self.dataframe)
        else:
            return self.dataframe

    @staticmethod
    def format_comma(x):
        """Format number with commas."""
        try:
            return "{:,.0f}".format(x)
        except ValueError:
            return x

    @staticmethod
    def format_million(x):
        """Format number in millions."""
        try:
            if x >= 1_000_000:
                return "${:.2f}M".format(x / 1_000_000)
            else:
                return "${:,.2f}".format(x)
        except ValueError:
            return x

    @staticmethod
    def format_percentage(x):
        """Format number as percentage."""
        try:
            return "{:.2f}%".format(x * 100)
        except ValueError:
            return x


# Usage and testing
if __name__ == "__main__":
    data = {
        'metric': ['new accounts', 'amount', 'credit loss'],
        't1': ["1,000", "5000000.0", "0.10"],
        't2': ["2,000", "6,000,000.0", "0.15"],
    }
    format_rules = {
        ('new accounts',): DataFrameFormatter.format_comma,
        ('amount',): DataFrameFormatter.format_million,
        ('credit loss',): DataFrameFormatter.format_percentage
    }

    df = pd.DataFrame(data)
    formatter = DataFrameFormatter(df, format_rules, metric_columns='metric')
    formatted_df = formatter.format_based_on_metric(inplace=False)
    print(formatted_df)

    # Just a dataframe with default formatting
    df_simple = pd.DataFrame({
        't1': ["1,000", "5,000,000.0", "0.10"],
        't2': ["2,000", "6,000,000.0", "0.15"],
    })
    formatter_simple = DataFrameFormatter(df_simple)
    formatted_df_simple = formatter_simple.apply_default_format(inplace=False)
    print(formatted_df_simple)
