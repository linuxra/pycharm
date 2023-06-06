import pandas as pd
from multiprocessing import Pool
import itertools

from concurrent.futures import ProcessPoolExecutor, as_completed

def get_last_8_quarters(date):
    """
    Get the last 8 quarters from a given date.

    Args:
    date: A string representing a date, like "2023-05-22".

    Returns:
    A list of strings representing the last 8 quarters, like ["t2021Q2", "t2021Q3", ...].
    """
    # Create a date range ending with the given date and spanning 8 quarters back
    date_range = pd.date_range(end=date, periods=8, freq='Q')

    # Convert the date range to quarters and format the quarters as strings
    quarters = ["t" + date.to_period('Q').strftime('%YQ%q') for date in date_range]

    # Return the quarters in reverse order (most recent first)
    return quarters

def function_1(*args):
    """
    Define the logic specific to row parameter 1.

    Args:
    i: The iterator value.
    col_param: The column parameter.

    Returns:
    The result of the computation.
    """
    i, col_param = args
    # Define your logic here, using 'i' as needed
    return 10

def get_quarter_dates(quarter_str):
    """
    Get the quarter begin and end dates from a string like "t2021Q1".

    Args:
    quarter_str: A string representing a quarter, like "t2021Q1".

    Returns:
    A tuple of the quarter begin and end dates as strings in 'yyyy-mm-dd' format.
    """
    # Extract the year and quarter from the string
    year, quarter = int(quarter_str[1:5]), int(quarter_str[6])

    # Create the quarter period
    quarter = pd.Period(f"{year}Q{quarter}")

    # Return the start and end dates of the quarter as strings
    return quarter.start_time.strftime('%Y-%m-%d'), quarter.end_time.strftime('%Y-%m-%d')
def function_2(*args):
    """
    Define the logic specific to row parameter 1.

    Args:
    i: The iterator value.
    col_param: The column parameter.

    Returns:
    The result of the computation.
    """
    i, col_param = args
    # Define your logic here, using 'i' as needed
    return 20


def function_3(*args):
    """
    Define the logic specific to row parameter 1.

    Args:
    i: The iterator value.
    col_param: The column parameter.

    Returns:
    The result of the computation.
    """
    i, col_param = args
    # Define your logic here, using 'i' as needed
    print(i)
    return 30


def function_4(*args):
    """
    Define the logic specific to row parameter 1.

    Args:
    i: The iterator value.
    col_param: The column parameter.

    Returns:
    The result of the computation.
    """
    i, col_param = args
    # Define your logic here, using 'i' as needed
    return 40


# ... Repeat for function_2, function_3, etc.

# Map row parameters to functions
function_dict = {
    1: function_1,
    2: function_2,
    3: function_3,
    4: function_4

    # ... Repeat for 2, 3, etc.
}


def cell_value_function(*args):
    """
    Get the specific function for the given row parameter and call it.

    Args:
    i: The iterator value.
    row_param: The row parameter.
    col_param: The column parameter.

    Returns:
    The result of the function call, or None if no function is found for the row parameter.
    """
    i, row_param, col_param = args
    function = function_dict.get(row_param, None)
    if function:
        return function(i, col_param)
    else:
        return None


# Define column names

# if __name__ == "__main__":
#     # Create a multiprocessing pool
#     with Pool(32) as p:
#         # Apply the function to each set of cell parameters in parallel
#         cell_values = p.map(cell_value_function, cell_parameters)
#     print(cell_values)
#     # Reshape the list of cell values into a 2D list representing the DataFrame
#     cell_values_2d = [cell_values[i:i + len(column_names) - 1] for i in
#                       range(0, len(cell_values), len(column_names) - 1)]
#
#     # Add the description column to the data
#     data = [[row_param] + cell_values for row_param, cell_values in zip(row_params, cell_values_2d)]
#
#     # Create the DataFrame
#     df = pd.DataFrame(data, columns=column_names)
#
#     # Your mapping dictionary
#     description_dict = {1: "Description 1", 2: "Description 2", 3: "Description 3", 4: "Description 4"}
#
#     # Replace the integers with their corresponding descriptions
#     df['Description'] = df['Description'].replace(description_dict)
#     print(df)
#     print(get_quarter_dates('t2021Q2'))
#     print(get_last_8_quarters('2023-04-21'))



def worker(*args):
    """
    Worker function to call the target function with the given arguments.

    Args:
    func: The target function to call.
    args: The arguments to pass to the target function.

    Returns:
    The return value of the target function.
    """
    func, args_list = args
    return func(args_list)

def parallel_execution1(func_list, args_list, num):
    """
    Execute the given functions in parallel using a process pool.

    Args:
    func_list: A list of target functions to execute.
    args_list: A list of arguments to pass to each function.
    num: The number of worker processes to use.

    Returns:
    A dictionary mapping each function's name to a list of result values returned by that function.
    """
    with ProcessPoolExecutor(max_workers=num) as executor:
        futures = {executor.submit(worker, func, args): func.__name__ for func, args in zip(func_list, args_list)}
        results_dict = {func.__name__: [] for func in func_list}
        for future in as_completed(futures):
            func_name = futures[future]
            try:
                result = future.result()
                results_dict[func_name].append(result)
                print(f"Task {func_name} completed successfully.")
            except Exception as exc:
                print(f"Task {func_name} generated an exception: {exc}")
    return results_dict

if __name__ == "__main__":
    column_names = ["Description", "t2021Q2", "t2021Q3", "t2021Q4", "t2022Q1", "t2022Q2", "t2022Q3", "t2022Q4",
                    "t2023Q1"]

    # Define row parameters (these should be your actual row parameters)
    row_params = [1, 2, 3, 4]

    # Create a list of tuples representing each cell's parameters
    cell_parameters = [(i, row_param, col_name) for i, (row_param, col_name) in
                       enumerate(itertools.product(row_params, column_names[1:]), 1)]
    print(cell_parameters)
    # Define the functions to execute
    functions = [function_1, function_2, function_3, function_4]

    # Define the arguments for each function
    args_list = cell_parameters  # This will execute each function with the same arguments

    # Execute the functions in parallel
    results_dict = parallel_execution1(functions, args_list, 32)

    # Print the results returned by each function
    for func_name, results in results_dict.items():
        print(f"{func_name}: {results}")

    # Reshape the list of cell values into a 2D list representing the DataFrame
    cell_values_2d = [results_dict[func_name] for func_name in functions]

    # Add the description column to the data
    data = [[row_param] + cell_values for row_param, cell_values in zip(row_params, cell_values_2d)]

    # Create the DataFrame
    df = pd.DataFrame(data, columns=column_names)

    # Your mapping dictionary
    description_dict = {1: "Description 1", 2: "Description 2", 3: "Description 3", 4: "Description 4"}

    # Replace the integers with their corresponding descriptions
    df['Description'] = df['Description'].replace(description_dict)
    print(df)
    print(get_quarter_dates('t2021Q2'))
    print(get_last_8_quarters('2023-04-21'))


