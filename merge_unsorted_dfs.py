import pandas as pd


def merge_dataframes(df_list):
    # Extract the 'yymm' part from the column names
    yymm_list = [col.split('_')[1] for df in df_list for col in df.columns if col.startswith('P_')]

    # Pair the 'yymm' values with the corresponding DataFrames
    df_yymm_pairs = zip(df_list, yymm_list)

    # Sort the list of DataFrames based on the 'yymm' values
    sorted_df_yymm_pairs = sorted(df_yymm_pairs, key=lambda x: x[1])

    # Extract the sorted DataFrames
    sorted_df_list = [pair[0] for pair in sorted_df_yymm_pairs]

    # Merge the sorted DataFrames one by one
    merged_df = sorted_df_list[0]

    for df in sorted_df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=['rank', 'counter'])

    return merged_df
