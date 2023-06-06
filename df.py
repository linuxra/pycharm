import pandas as pd
import numpy as np


def generate_score_bands(start=300, end=850, num_bands=10, counter_size=24, score_bands=None, input_df=None):
    """
    Generates a dataframe with FICO score bands, their rank, and a counter.

    Parameters:
    start (int, optional): The start of the FICO score range. Default is 300.
    end (int, optional): The end of the FICO score range. Default is 850.
    num_bands (int, optional): The number of score bands to create. Default is 10.
    counter_size (int, optional): The size of the counter for each score band. Default is 24.
    score_bands (tuple, optional): A tuple of score bands to use instead of generating them. Overrides start, end, and num_bands.
    input_df (pd.DataFrame, optional): An input DataFrame containing a 'FICO_Score' column. If provided, score bands will be based on the distribution of FICO scores in the input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with columns 'ScoreRange', 'Rank', 'Counter'.
    """
    if input_df is not None:
        # Calculate quantiles based on the distribution of FICO scores
        score_bands = input_df['FICO_Score'].quantile(np.linspace(0, 1, num_bands + 1)).tolist()
    elif score_bands is None:
        # Generate score bands if not provided
        score_bands = np.linspace(start, end, num_bands + 1).tolist()
    else:
        # Use the provided score_bands
        score_bands = sorted(list(score_bands))

    # Generate counters
    counters = list(range(1, counter_size + 1))

    # Create a list to store the data
    data = []

    # Generate the data
    for i in range(len(score_bands) - 1):
        for counter in counters:
            score_range = f"{round(score_bands[i], 2)}-{round(score_bands[i + 1], 2)}"
            data.append([score_range, i + 1, counter])

    # Create a dataframe from the data
    df = pd.DataFrame(data, columns=['ScoreRange', 'Rank', 'Counter'])

    return df


# Test the function with an input DataFrame
input_data = {'FICO_Score': [350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
input_df = pd.DataFrame(input_data)
df3 = generate_score_bands(input_df=input_df)

print(df3)
import pandas as pd

def merge_dataframes(left_df, right_df, right_columns, keys):
    """
    Merge two dataframes on multiple keys. Keep all columns from the left dataframe
    and only the specified columns from the right dataframe.

    Parameters:
    left_df (pd.DataFrame): The left dataframe.
    right_df (pd.DataFrame): The right dataframe.
    right_columns (List[str]): The columns to keep from the right dataframe.
    keys (List[str]): The columns to merge on.

    Returns:
    pd.DataFrame: The merged dataframe.
    """
    # Keep only the specified columns from the right dataframe
    right_df = right_df[right_columns + keys]

    # Merge the dataframes
    merged_df = pd.merge(left_df, right_df, on=keys, how='left')

    return merged_df

