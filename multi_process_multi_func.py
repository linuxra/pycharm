from table4 import Table
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta



def f1(start_date, end_date, counter):

    df = pd.DataFrame({"Function Name": ['f1'],
                       "Start Date": [start_date],
                       "End Date": [end_date],
                       "Counter": [counter]})
    return df


def f2(start_date, end_date, counter):

    df = pd.DataFrame({"Function Name": ['f2'],
                       "Start Date": [start_date],
                       "End Date": [end_date],
                       "Counter": [counter]})
    return df


def generate_dates(n_months):
    start_date = datetime.now()
    return [
        (start_date + relativedelta(months=i), start_date + relativedelta(months=i + 1) - relativedelta(days=1), i + 1)
        for i in range(n_months)]


def worker(func, dates):
    return func(*dates)


def parallel_execution(func_list, dates_list):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, func, dates): (func.__name__, dates) for func in func_list for dates in
                   dates_list}
        results = []
        for future in as_completed(futures):
            func_name, dates = futures[future]
            try:
                df = future.result()
                results.append(df)
                print(f"Task {func_name} with dates {dates} completed successfully.")
            except Exception as exc:
                print(f"Task {func_name} with dates {dates} generated an exception: {exc}")
    return results

def parallel_execution1(func_list, dates_list, num):
    with ProcessPoolExecutor(max_workers = num ) as executor:
        futures = {executor.submit(worker, func, dates): func.__name__ for func in func_list for dates in dates_list}
        dfs_dict = {func.__name__: [] for func in func_list}
        for future in as_completed(futures):
            func_name = futures[future]
            try:
                df = future.result()
                dfs_dict[func_name].append(df)
                print(f"Task {func_name} completed successfully.")
            except Exception as exc:
                print(f"Task {func_name} generated an exception: {exc}")
    return {func_name: pd.concat(dfs) for func_name, dfs in dfs_dict.items()}


def parallel_execution2(func_list, dates_list, num):
    """
    Perform parallel execution of a list of functions with corresponding lists of dates.

    Args:
        func_list (list): List of functions to execute.
        dates_list (list): List of lists of dates corresponding to each function.
        num (int): Maximum number of worker processes.

    Returns:
        dict: A dictionary where each function name is associated with a list of results.

    """
    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num) as executor:
        # Submit tasks and store futures with function names
        futures = {executor.submit(worker, func, dates): func.__name__
                   for func, date_list in zip(func_list, dates_list)
                   for dates in date_list}

        # Initialize an empty dictionary to store results
        dfs_dict = {func.__name__: [] for func in func_list}

        # Process completed futures as they finish
        for future in as_completed(futures):
            func_name = futures[future]
            try:
                # Get the result of the task
                result = future.result()
                dfs_dict[func_name].append(result)
                print(f"Task {func_name} completed successfully.")
            except Exception as exc:
                # Handle exceptions that occurred during task execution
                print(f"Task {func_name} generated an exception: {exc}")

    return dfs_dict


if __name__ == '__main__':
    # Define the list of functions
    functions = [f1, f2]

    # Generate the list of dates
    dates = generate_dates(36)

    # Execute the functions in parallel
    result_dfs = parallel_execution1(functions, dates, 6)

    for func_name, df in result_dfs.items():
        print(f"Dataframe for function {func_name}:")
        print(df)
        globals()[f"df_{func_name}"] = df

    table1 = Table(df_f1,"First Table")
    table1.save("First_Table.pdf")

    table2 = Table(df_f2,"second_table")
    table2.save("second_table.pdf")