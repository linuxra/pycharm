import multiprocessing
import time
from typing import List, Callable, Any, Tuple


# This function is used to execute multiple functions in parallel using multiprocessing.
def multiprocess_functions(
    functions: List[Callable],
    params: List[Tuple[Any, ...]],
    num_processes: int,
    timeout: float = None,
    progress_bar: bool = False,
) -> List[Any]:
    """
    Execute multiple functions in parallel using multiprocessing.

    Args:
        functions (List[Callable]): A list of callable functions to be executed.
        params (List[Tuple[Any, ...]]): A list of tuples, where each tuple contains arguments for the corresponding function.
        num_processes (int): The number of processes to use for executing the functions.
        timeout (float, optional): The maximum time in seconds that each function can run. Defaults to None, which means no timeout.
        progress_bar (bool, optional): If True, displays a progress bar using tqdm. Defaults to False.

    Returns:
        List[Any]: A list of results from the function executions, or errors if they occur.
    """
    # Check if the functions and params are lists
    if not isinstance(functions, list) or not isinstance(params, list):
        raise TypeError("Both functions and params must be lists")
    # Check if the number of functions and parameters match
    if len(functions) != len(params):
        raise ValueError("The number of functions and parameters must match")
    # Check if the number of processes is a positive integer
    if not isinstance(num_processes, int) or num_processes <= 0:
        raise TypeError("The number of processes must be a positive integer")
    # Check if all elements in the functions list are callable
    if not all(callable(func) for func in functions):
        raise TypeError("All elements in the functions list must be callable")

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results: list[Any] = []
        # For each function and its parameters, apply them asynchronously
        for func, param in zip(functions, params):
            result = pool.apply_async(func, param)
            results.append(result)

        output = []
        # If progress bar is True, import tqdm and wrap the results with it
        if progress_bar:
            from tqdm import tqdm

            results = tqdm(results)

        # Collect results from the asynchronous operations
        for i, result in enumerate(results):
            try:
                # Get the result of the function execution
                res = result.get(timeout=timeout)
                output.append(res)
            except multiprocessing.TimeoutError:
                # If the function execution times out, append an error message
                output.append(f"Function {functions[i].__name__} timed out")
            except Exception as e:
                # If an error occurs during function execution, append an error message
                output.append(f"Error in function {functions[i].__name__}: {str(e)}")

        return output


# This function calculates the square of a number.
def square_number(n: int) -> int:
    """
    Returns the square of a number.

    Args:
        n (int): The number to be squared.

    Returns:
        int: The square of the number.

    Note:
        This function simulates a delay of 10 seconds before returning the result.
    """
    time.sleep(10)  # Simulate a delay
    return n * n


# This function returns the input string repeated twice.
def repeat_string(s: str) -> str:
    """
    Returns the input string repeated twice.

    Args:
        s (str): The string to be repeated.

    Returns:
        str: The input string repeated twice.

    Note:
        This function simulates a delay of 20 seconds before returning the result.
    """
    time.sleep(20)  # Simulate a longer delay
    return s + s


# This function calculates the factorial of a number.
def factorial(n: int) -> int:
    """
    Calculates the factorial of a number.

    Args:
        n (int): The number to calculate the factorial of.

    Returns:
        int: The factorial of the number.
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# Example use case
if __name__ == "__main__":
    import multiprocessing

    # Set multiprocessing start method to 'spawn' for compatibility, especially on Windows
    multiprocessing.set_start_method("spawn")

    # Define the functions to be executed
    functions = [square_number, repeat_string, factorial]
    # Define the parameters for the functions
    params = [(5,), ("hello",), (6,)]

    # Execute the functions in parallel and get the results
    results = multiprocess_functions(
        functions=functions,
        params=params,
        num_processes=3,
        timeout=30,
        progress_bar=True,
    )

    # Print the results
    print("Results:", results)
