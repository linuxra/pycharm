import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from log_1 import logging_decorator

import pandas as pd
from sqlalchemy import create_engine
import logging

import logging
from queries import Queries
from another_class import AnotherClass

# Set up the logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@logging_decorator(logger)
def teradata_query(param1, param2):
    """
    Function that activates the Teradata connection, runs two queries and returns a dataframe.

    Args:
    param1: First parameter to be used in the queries.
    param2: Second parameter to be used in the queries.

    Returns:
    A Pandas DataFrame containing the results of the queries.
    """
    engine = create_engine('postgresql://postgres:temp123@localhost/postgres')

    # Save the DataFrame to the PostgresSQL table
    print({param1}, {param2})
    df = pd.read_sql('SELECT * from soccer_matches limit 10', engine)

    return df


@logging_decorator(logger)
def run_queries_concurrently(func, params: object):
    """
    Function to run the given function concurrently with the provided parameters.

    Args:
    func: A function to be executed concurrently.
    params: A list of tuples containing the parameters for each function call.

    Returns:
    Merged DataFrame of the results from all function calls.
    :param func:
    :type params: object
    """

    dataframes = []



    # Use ThreadPoolExecutor to efficiently run the teradata_query function with a maximum of 6 threads
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit( func, param1, param2) for param1, param2 in params]

        for future in as_completed(futures):
            try:
                df = future.result()
                if df is not None:
                    dataframes.append(df)
            except Exception as e:
                print(f"Error occurred in the thread: {e}")

    # Merge all returned dataframes
    merged_dataframe = pd.concat(dataframes, ignore_index=True)
    return merged_dataframe


if __name__ == "__main__":
    # Define the parameters for each function call (total of 36)
    params = [
        ('p1', 'p2'), ('p1', 'p2')  # Replace with actual parameter values
        # Add more tuples for a total of 36
    ]

    # Call run_queries_concurrently with the teradata_query function and the list of parameters
    merged_dataframe = run_queries_concurrently(teradata_query, params)
    print(merged_dataframe)
