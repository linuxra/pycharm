import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_queries_concurrently(func, params: object):
    """
    Function to run the given function concurrently with the provided parameters.

    Args:
    func: A function to be executed concurrently.
    params: A list of tuples containing the parameters for each function call.

    Returns:
    Merged DataFrame of the results from all function calls.
    """

    dataframes = []

    # Use ThreadPoolExecutor to efficiently run the given function with a maximum of 6 threads
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(func, *param) for param in params]

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
import pandas as pd
import numpy as np

# Sample function 1
def function_1(rows: int, cols: int):
    data = np.random.randint(0, 100, size=(rows, cols))
    columns = [f'col_{i}' for i in range(cols)]
    return pd.DataFrame(data, columns=columns)

# Sample function 3
def function_3(letters: int, numbers: int, rows: int):
    data = [list(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=letters)) +
            list(np.random.randint(1, 100, size=numbers)) for _ in range(rows)]
    return pd.DataFrame(data, columns=[f'char_{i}' for i in range(letters)] + [f'num_{i}' for i in range(numbers)])

# Prepare the list of tuples containing parameters for each function call
params = [
    (3, 5),            # Parameters for function_1
    (4, 2, 10)         # Parameters for function_3
]

# Run the functions concurrently
merged_dataframe = run_queries_concurrently(function_1, params[:1])
print("Merged DataFrame for function_1:\n", merged_dataframe)

merged_dataframe = run_queries_concurrently(function_3, params[1:])
print("\nMerged DataFrame for function_3:\n", merged_dataframe)
