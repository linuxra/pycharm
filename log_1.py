# # logger.py
#
# import functools
# import inspect
# import pandas as pd
#
#
# def logging_decorator(logger=None):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             first_arg = args[0] if args else None
#
#             # Determine whether the logger should be obtained from the class instance
#             if logger is None and first_arg and hasattr(first_arg, "logger"):
#                 instance_logger = first_arg.logger
#             else:
#                 instance_logger = logger
#
#             def arg_repr(arg):
#                 if isinstance(arg, pd.DataFrame):
#                     return f"DataFrame: \n{arg.head()}"
#                 elif isinstance(arg, (str, int, float)):
#                     return str(arg)
#                 elif isinstance(arg, (list, tuple)):
#                     return f"{arg.__class__.__name__}({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
#                 elif isinstance(arg, dict):
#                     return f"dict({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
#                 else:
#                     return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"
#
#             args_repr = ', '.join(arg_repr(arg) for arg in args)
#             kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())
#
#             try:
#                 instance_logger.info(
#                     f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
#                 result = func(*args, **kwargs)
#                 instance_logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
#                 return result
#             except Exception as e:
#                 instance_logger.error(f"Function '{func.__name__}' raised an exception: {e}")
#                 raise
#
#         return wrapper
#     return decorator


import functools
import inspect
import pandas as pd
import os
import psutil
import time
from typing import Any, Callable, Optional


def logging_decorator(logger: Optional = None) -> Callable:
    """
    A decorator to log information about the execution of decorated functions.

    This decorator logs the function call with its arguments and the returned result.
    It handles various argument types like DataFrame, string, int, float, list, tuple, and dict
    by providing informative and truncated string representations.

    Additionally, this decorator logs memory usage before and after the function call, as well as the time it took to execute
    the function.

    Args:
        logger: A logger object responsible for logging messages.

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:

            first_arg = args[0] if args else None

            # Determine whether the logger should be obtained from the class instance
            if logger is None and first_arg and hasattr(first_arg, "logger"):
                instance_logger = first_arg.logger
            else:
                instance_logger = logger

            def arg_repr(arg: Any) -> str:
                if isinstance(arg, pd.DataFrame):
                    return f"DataFrame: \n{arg.head()}"
                elif isinstance(arg, (str, int, float)):
                    return str(arg)
                elif isinstance(arg, (list, tuple)):
                    return f"{arg.__class__.__name__}({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
                elif isinstance(arg, dict):
                    return f"dict({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
                else:
                    return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                instance_logger.info(
                    f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")

                # Log memory usage before the function call
                process = psutil.Process(os.getpid())
                mem_info_before = process.memory_info()

                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()

                logger_message = f"Function '{func.__name__}' took {end_time - start_time:.2f} seconds to execute"
                instance_logger.info(logger_message)

                logger_message = f"Function '{func.__name__}' returned: {arg_repr(result)}"
                instance_logger.info(logger_message)

                # Log memory usage after the function call
                mem_info_after = process.memory_info()
                mem_diff = mem_info_after.rss - mem_info_before.rss

                logger_message = f"Function '{func.__name__}' memory usage increased by {mem_diff} bytes"
                instance_logger.info(logger_message)

                return result
            except Exception as e:
                instance_logger.error(f"Function '{func.__name__}' raised an exception: {e}")
                raise

        return wrapper

    return decorator
