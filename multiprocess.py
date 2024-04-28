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
    if not isinstance(functions, list) or not isinstance(params, list):
        raise TypeError("Both functions and params must be lists")
    if len(functions) != len(params):
        raise ValueError("The number of functions and parameters must match")
    if not isinstance(num_processes, int) or num_processes <= 0:
        raise TypeError("The number of processes must be a positive integer")
    if not all(callable(func) for func in functions):
        raise TypeError("All elements in the functions list must be callable")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for func, param in zip(functions, params):
            result = pool.apply_async(func, param)
            results.append(result)

        output = []
        if progress_bar:
            from tqdm import tqdm

            results = tqdm(results)

        for i, result in enumerate(results):
            try:
                res = result.get(timeout=timeout)
                output.append(res)
            except multiprocessing.TimeoutError:
                output.append(f"Function {functions[i].__name__} timed out")
            except Exception as e:
                output.append(f"Error in function {functions[i].__name__}: {str(e)}")

        return output


def square_number(n):
    import time

    time.sleep(10)  # Simulate a delay
    return n * n


def repeat_string(s):
    import time

    time.sleep(20)  # Simulate a longer delay
    return s + s


def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


functions = [square_number, repeat_string, factorial]
params = [(5,), ("hello",), (6,)]

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")  # Necessary on Windows

    results = multiprocess_functions(
        functions=functions,
        params=params,
        num_processes=3,
        timeout=30,
        progress_bar=True,
    )

    print("Results:", results)
