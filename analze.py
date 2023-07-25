import pandas as pd
import numpy as np

# Create a DataFrame with dummy data
data = {
    'Metric': ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5', 'Metric 6', 'Metric 7'],
    '2021Q1': ['100', '200', '300', '400', '500', '600', '700'],
    '2021Q2': ['110', '210', '310', '410', '510', '610', '710'],
    '2021Q3': ['120', '220', '320', '420', '520', '620', '720'],
    '2021Q4': ['130', '230', '330', '430', '530', '630', '730'],
    '2022Q1': ['140', '240', '340', '440', '540', '640', '740'],
    '2022Q2': ['150', '250', '350', '450', '550', '650', '750'],
    '2022Q3': ['160', '260', '360', '460', '560', '660', '760'],
    '2022Q4': ['170', '270', '370', '470', '570', '670', '770'],
}

df = pd.DataFrame(data)

def analyze_metrics(df):
    # Initialize an empty string to hold the markdown text
    markdown_text = ""

    # Convert all columns (except 'Metric') to numeric, non-numeric values are converted to NaN
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        metric = row['Metric']

        # Calculate basic statistics
        min_val = row[1:].min()
        max_val = row[1:].max()

        # Manually find the index of the min and max values
        min_period = row.index[np.where(row[1:] == min_val)[0][0] + 1]
        max_period = row.index[np.where(row[1:] == max_val)[0][0] + 1]

        # Calculate simple trend
        start_val = row[1]
        end_val = row[-1]
        if start_val < end_val:
            trend = "increased"
        elif start_val > end_val:
            trend = "decreased"
        else:
            trend = "stayed the same"

        # Calculate QoQ growth
        qoq_growth = row[1:].pct_change()

        # Add analysis for this metric to the markdown text
        markdown_text += f"""
## Analysis for {metric}

- The value for {metric} ranged from {min_val} in {min_period} to {max_val} in {max_period}.
- The period with the highest value for {metric} was {max_period} with {max_val}.
- The period with the lowest value for {metric} was {min_period} with {min_val}.
- Overall, the value for {metric} has {trend} from {start_val} in the first period to {end_val} in the last period.
- The QoQ growth for {metric} ranged from {qoq_growth.min()} to {qoq_growth.max()}.
        """

    # Calculate utilization ratio
    row_4 = df.iloc[3, 1:]  # 0-indexed, so row 4 is at index 3
    row_5 = df.iloc[4, 1:]  # 0-indexed, so row 5 is at index 4
    utilization_ratio = row_5 / row_4
    min_ratio = utilization_ratio.min()
    max_ratio = utilization_ratio.max()

    # Manually find the index of the min and max values
    min_period_ratio = utilization_ratio.index[np.where(utilization_ratio == min_ratio)[0][0]]
    max_period_ratio = utilization_ratio.index[np.where(utilization_ratio == max_ratio)[0][0]]

    # Add analysis for utilization ratio to the markdown text
    markdown_text += f"""
## Analysis for Utilization Ratio

- The utilization ratio ranged from {min_ratio} in {min_period_ratio} to {max_ratio} in {max_period_ratio}.
- The period with the highest utilization ratio was {max_period_ratio} with {max_ratio}.
- The period with the lowest utilization ratio was {min_period_ratio} with {min_ratio}.
    """

    # Return the complete markdown text
    return markdown_text

# Call the function and print the result
markdown_text = analyze_metrics(df)
print(markdown_text)
