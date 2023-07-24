import pandas as pd
import numpy as np


class PSICalculator:
    def __init__(self, baseline_df, current_df):
        self.baseline_df = baseline_df
        self.current_df = current_df
        self.psi_df = None

    def calculate_psi(self):
        # Merge the two dataframes on rank, aa_code, and code_value
        psi_df = pd.merge(self.baseline_df, self.current_df, on=['rank', 'aa_code', 'code_value'],
                          suffixes=('_base', '_current'))

        # Replace 0s in the percent columns to avoid division by zero
        psi_df.replace({0: 0.000001}, inplace=True)

        # Calculate the PSI
        psi_df['psi'] = (psi_df['percent_current'] - psi_df['percent_base']) * np.log(
            psi_df['percent_current'] / psi_df['percent_base'])

        # Calculate total PSI for each rank and aa_code
        psi_df['total_psi'] = psi_df.groupby(['rank', 'aa_code'])['psi'].transform('sum')

        self.psi_df = psi_df

        return psi_df


# Example usage:

# Create some example data
baseline_df = pd.DataFrame({
    'rank': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'aa_code': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    'code_value': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
    'base_freq': [100, 120, 200, 180, 150, 170, 250, 230, 300, 320],
    'percent_base': [0.1, 0.12, 0.2, 0.18, 0.15, 0.17, 0.25, 0.23, 0.3, 0.32]
})

current_df = pd.DataFrame({
    'rank': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'aa_code': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    'code_value': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
    'current_freq': [120, 130, 180, 190, 160, 170, 240, 250, 300, 310],
    'percent_current': [0.12, 0.13, 0.18, 0.19, 0.16, 0.17, 0.24, 0.25, 0.3, 0.31]
})

# Calculate PSI
psi_calculator = PSICalculator(baseline_df, current_df)
psi_df = psi_calculator.calculate_psi()

print("Complete PSI Dataframe:\n", psi_df)

