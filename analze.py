def analyze_metrics(df):
    # Initialize an empty string to hold the markdown text
    markdown_text = ""

    # Iterate over each column in the dataframe (excluding the first column which is assumed to be dates/periods)
    for metric in df.columns[1:]:
        # Calculate basic statistics
        min_val = df[metric].min()
        max_val = df[metric].max()
        min_period = df[df[metric] == min_val].index[0]
        max_period = df[df[metric] == max_val].index[0]

        # Calculate simple trend
        start_val = df.loc[df.index[0], metric]
        end_val = df.loc[df.index[-1], metric]
        if start_val < end_val:
            trend = "increased"
        elif start_val > end_val:
            trend = "decreased"
        else:
            trend = "stayed the same"

        # Calculate QoQ growth
        qoq_growth = df[metric].pct_change()

        # Add analysis for this metric to the markdown text
        markdown_text += f"""
## Analysis for {metric}

- The value for {metric} ranged from {min_val} in {min_period} to {max_val} in {max_period}.
- The period with the highest value for {metric} was {max_period} with {max_val}.
- The period with the lowest value for {metric} was {min_period} with {min_val}.
- Overall, the value for {metric} has {trend} from {start_val} in {df.index[0]} to {end_val} in {df.index[-1]}.
- The QoQ growth for {metric} ranged from {qoq_growth.min()} to {qoq_growth.max()}.
        """

    # Calculate utilization ratio
    row_4 = df.iloc[3, 1:]  # 0-indexed, so row 4 is at index 3
    row_5 = df.iloc[4, 1:]  # 0-indexed, so row 5 is at index 4
    utilization_ratio = row_5 / row_4
    min_ratio = utilization_ratio.min()
    max_ratio = utilization_ratio.max()
    min_period_ratio = utilization_ratio.idxmin()
    max_period_ratio = utilization_ratio.idxmax()

    # Add analysis for utilization ratio to the markdown text
    markdown_text += f"""
## Analysis for Utilization Ratio

- The utilization ratio ranged from {min_ratio} in {min_period_ratio} to {max_ratio} in {max_period_ratio}.
- The period with the highest utilization ratio was {max_period_ratio} with {max_ratio}.
- The period with the lowest utilization ratio was {min_period_ratio} with {min_ratio}.
    """

    # Return the complete markdown text
    return markdown_text


# Load the csv file
df = pd.read_csv('path_to_your_file/accts.csv')

# Call the function and print the result
markdown_text = analyze_metrics(df)
print(markdown_text)
