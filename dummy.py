import pandas as pd

# Create DataFrames with rank and counter
rank_df = pd.DataFrame({'rank': list(range(1, 11))})
counter_df = pd.DataFrame({'counter': list(range(1, 25))})

# Add a common key column to both DataFrames for the cross join
rank_df['key'] = 1
counter_df['key'] = 1

# Perform the cross join
cross_joined_df = pd.merge(rank_df, counter_df, on='key').drop(columns='key')

print(cross_joined_df)
