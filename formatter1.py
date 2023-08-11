import pandas as pd


class DataFrameFormatter:
    def __init__(self, dataframe, format_rules=None, metric_columns=None, columns_to_format=None):
        self.dataframe = dataframe.copy()
        self.format_rules = format_rules or {}

        if metric_columns:
            if isinstance(metric_columns, str):
                self.metric_columns = [metric_columns]
            else:
                self.metric_columns = metric_columns
            excluded_cols = set(self.metric_columns)
            self.columns_to_format = columns_to_format or [col for col in dataframe.columns if col not in excluded_cols]
        else:
            self.metric_columns = None
            self.columns_to_format = columns_to_format or dataframe.columns

    @staticmethod
    def _convert_to_number(value):
        """Converts a string to a number if possible."""
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def apply_default_format(self):
        for col in self.columns_to_format:
            self.dataframe[col] = self.dataframe[col].apply(self._convert_to_number).apply(self.format_comma)
        return self

    def apply_format(self, format_type):
        if format_type == "comma":
            formatter = self.format_comma
        elif format_type == "million":
            formatter = self.format_million
        elif format_type == "percentage":
            formatter = self.format_percentage
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        for col in self.columns_to_format:
            self.dataframe[col] = self.dataframe[col].apply(self._convert_to_number).apply(formatter)
        return self

    def format_based_on_metric(self):
        if not self.metric_columns:
            raise ValueError("metric_columns must be provided for this method.")

        for idx, row in self.dataframe.iterrows():
            metric_keys = tuple(row[col] for col in self.metric_columns)
            formatter = self.format_rules.get(metric_keys)
            if formatter:
                for col in self.columns_to_format:
                    value = row[col]
                    if isinstance(value, str):
                        value = self._convert_to_number(value)
                    self.dataframe.at[idx, col] = formatter(value)
        return self

    def get_formatted_dataframe(self):
        return self.dataframe

    @staticmethod
    def format_comma(x):
        return "{:,.0f}".format(x)

    @staticmethod
    def format_million(x):
        if x >= 1_000_000:
            return "${:.2f}M".format(x / 1_000_000)
        else:
            return "${:,.2f}".format(x)

    @staticmethod
    def format_percentage(x):
        return "{:.2f}%".format(x * 100)


# Usage example
if __name__ == "__main__":
    data = {
        'metric': ['new accounts', 'amount', 'credit loss'],
        'sub_metric': ['a', 'b', 'c'],
        't1': ["1000", "5000000.0", "0.10"],
        't2': ["2000", "6000000.0", "0.15"],
    }

    format_rules = {
        ('new accounts', 'a'): DataFrameFormatter.format_comma,
        ('amount', 'b'): DataFrameFormatter.format_million,
        ('credit loss', 'c'): DataFrameFormatter.format_percentage
    }

    df = pd.DataFrame(data)
    formatter = DataFrameFormatter(df, format_rules, metric_columns=['metric', 'sub_metric'])
    formatted_df = formatter.format_based_on_metric().get_formatted_dataframe()
    print(formatted_df)
    data_simple = {
        't1': ["1000", "5000000.0", "0.10"],
        't2': ["2000", "6000000.0", "0.15"],
    }

    df_simple = pd.DataFrame(data_simple)
    formatter_simple = DataFrameFormatter(df_simple)
    formatted_df_simple = formatter_simple.apply_default_format().get_formatted_dataframe()
    print(formatted_df_simple)
