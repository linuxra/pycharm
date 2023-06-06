import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def generate_case_statement_and_ranges(df: pd.DataFrame, score_column: str, num_ranges: int,
                                       output_column_name: str = 'score_rank') -> Tuple[str, str, pd.DataFrame]:
    """
    Generate two SQL case statements and a DataFrame with low, high, and rank columns based on the distribution
    of the data in a DataFrame. One case statement assigns a rank based on the score range, and the other assigns
    the score range as a string.

    :param df: Input DataFrame containing the scores
    :param score_column: Name of the column containing the scores
    :param num_ranges: Number of desired score ranges
    :param output_column_name: Name of the output column in the case statement (default is 'score_rank')
    :return: Two SQL case statement strings and a DataFrame with low, high, and rank columns
    """
    quantiles = np.linspace(0, 1, num_ranges + 1)
    cutoff_points = df[score_column].quantile(quantiles).values

    case_statement_rank = f"CASE"
    case_statement_range = f"CASE"

    range_data = []

    for i in range(len(cutoff_points) - 1):
        low = int(np.ceil(cutoff_points[i]))
        high = int(np.floor(cutoff_points[i + 1]))
        case_statement_rank += f" WHEN {score_column} BETWEEN {low} AND {high} THEN {i + 1}"
        case_statement_range += f" WHEN {score_column} BETWEEN {low} AND {high} THEN '{low}-{high}'"

        range_data.append({'low': low, 'high': high, 'rank': i + 1})

    case_statement_rank += f" ELSE NULL END AS {output_column_name}"
    case_statement_range += f" ELSE NULL END AS score_range"

    ranges_df = pd.DataFrame(range_data)

    return case_statement_rank, case_statement_range, ranges_df


num_ranges = 4
score_column = 'fico_score'

sql_case_statement_rank, sql_case_statement_range, ranges_df = generate_case_statement_and_ranges(input_df,
                                                                                                  score_column,
                                                                                                  num_ranges)

print(sql_case_statement_rank)
print(sql_case_statement_range)
print(ranges_df)
