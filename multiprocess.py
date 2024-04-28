import multiprocessing
import time
from typing import List, Callable, Any, Tuple


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
    if not isinstance(functions, list) or not isinstance(params, list):
        raise TypeError("Both functions and params must be lists")
    if len(functions) != len(params):
        raise ValueError("The number of functions and parameters must match")
    if not isinstance(num_processes, int) or num_processes <= 0:
        raise TypeError("The number of processes must be a positive integer")
    if not all(callable(func) for func in functions):
        raise TypeError("All elements in the functions list must be callable")

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for func, param in zip(functions, params):
            result = pool.apply_async(func, param)
            results.append(result)

        output = []
        if progress_bar:
            from tqdm import tqdm

            results = tqdm(results)

        # Collect results from the asynchronous operations
        for i, result in enumerate(results):
            try:
                res = result.get(timeout=timeout)
                output.append(res)
            except multiprocessing.TimeoutError:
                output.append(f"Function {functions[i].__name__} timed out")
            except Exception as e:
                output.append(f"Error in function {functions[i].__name__}: {str(e)}")

        return output


def square_number(n: int) -> int:
    """Returns the square of a number."""
    time.sleep(10)  # Simulate a delay
    return n * n


def repeat_string(s: str) -> str:
    """Returns the input string repeated twice."""
    time.sleep(20)  # Simulate a longer delay
    return s + s


def factorial(n: int) -> int:
    """Calculates the factorial of a number."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# Example use case
if __name__ == "__main__":
    import multiprocessing

    # Set multiprocessing start method to 'spawn' for compatibility, especially on Windows
    multiprocessing.set_start_method("spawn")

    functions = [square_number, repeat_string, factorial]
    params = [(5,), ("hello",), (6,)]

    results = multiprocess_functions(
        functions=functions,
        params=params,
        num_processes=3,
        timeout=30,
        progress_bar=True,
    )

    print("Results:", results)
