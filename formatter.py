import pandas as pd

class DataFrameFormatter:
    def __init__(self, dataframe, format_rules):
        """
        :param dataframe: The pandas DataFrame to format.
        :param format_rules: A dictionary with metric names as keys and format functions as values.
        """
        self.dataframe = dataframe
        self.format_rules = format_rules

    def format_based_on_metric(self):
        time_columns = [col for col in self.dataframe.columns if col.startswith('t')]

        for idx, row in self.dataframe.iterrows():
            metric = row['metric']
            formatter = self.format_rules.get(metric)
            if formatter:
                for time_col in time_columns:
                    value = row[time_col]
                    self.dataframe.at[idx, time_col] = formatter(value)

        return self

    def get_formatted_dataframe(self):
        return self.dataframe

# Define your custom format functions
def format_comma(x):
    return "{:,.0f}".format(x)

def format_million(x):
    if x >= 1_000_000:
        return "${:.2f}M".format(x / 1_000_000)
    else:
        return "${:,.2f}".format(x)

def format_percentage(x):
    return "{:.2f}%".format(x)

# Usage example
if __name__ == "__main__":
    data = {
        'metric': ['new accounts', 'amount', 'credit loss'],
        't1': [1000.0, 5000000.0, 10.0],
        't2': [2000.0, 6000000.0, 15.0],
        # ... and so on for t3, t4, etc.
    }
    format_rules = {
        'new accounts': format_comma,
        'amount': format_million,
        'credit loss': format_percentage
    }

    df = pd.DataFrame(data)
    formatter = DataFrameFormatter(df, format_rules)
    formatted_df = formatter.format_based_on_metric().get_formatted_dataframe()
    print(formatted_df)

