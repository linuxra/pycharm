import functools
import pandas as pd
import numpy as np
import time
import os
import psutil
from typing import Any, Callable, Optional


class LoggingDecorator:
    def __init__(self, func: Callable) -> None:
        """
        Initializes the decorator with the function to be decorated.

        Args:
            func (Callable): The function to be decorated.
        """
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, instance: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the decorated function and logs its call and return information.

        Args:
            instance: The class instance on which the decorated method is called.
            *args: Positional arguments passed to the decorated function.
            **kwargs: Keyword arguments passed to the decorated function.

        Returns:
            The return value of the decorated function.
        """
        logger = getattr(instance, "logger", None)
        if logger is not None:
            logger.info(f"Calling function '{self.func.__name__}' with args: {args} and kwargs: {kwargs}")

        start_time = time.time()

        try:
            result = self.func(instance, *args, **kwargs)
        except Exception as e:
            if logger is not None:
                logger.error(f"Exception occurred while calling '{self.func.__name__}': {e}")
            raise
        else:
            end_time = time.time()
            elapsed_time = end_time - start_time

            if logger is not None:
                logger.info(f"Function '{self.func.__name__}' took {elapsed_time:.2f} seconds to execute")

                if isinstance(result, pd.DataFrame):
                    shape = result.shape
                    logger.info(f"Function '{self.func.__name__}' returned a DataFrame with shape {shape}")

                    head_obs = kwargs.get('head_obs')
                    if head_obs is not None:
                        head = result.head(head_obs)
                        logger.info(f"DataFrame head:\n{head}")

                elif isinstance(result, np.ndarray):
                    shape = result.shape
                    logger.info(f"Function '{self.func.__name__}' returned a NumPy array with shape {shape}")

                elif isinstance(result, (list, tuple, set, dict)):
                    length = len(result)
                    if length < 100:  # You can adjust this limit based on your preferences
                        logger.info(f"Function '{self.func.__name__}' returned: {result}")
                    else:
                        logger.info(f"Function '{self.func.__name__}' returned a collection of length {length}")

                elif hasattr(result, '__class__'):
                    class_name = result.__class__.__name__
                    logger.info(f"Function '{self.func.__name__}' returned an object of class '{class_name}'")

                else:
                    logger.info(f"Function '{self.func.__name__}' returned: {result}")

                # Log memory usage
                process = psutil.Process(os.getpid())
                mem_info_before = process.memory_info()

                mem_info_after = process.memory_info()
                mem_diff = mem_info_after.rss - mem_info_before.rss

                logger.info(f"Function '{self.func.__name__}' memory usage increased by {mem_diff} bytes")

            return result

    def __get__(self, instance: Any, owner: type) -> Callable:
        """
        Returns a functools.partial object that has the class instance pre-filled as
        the first argument for the __call__ method.

        Args:
            instance: The class instance on which the decorated method is accessed.
            owner: The owner
Returns:
        A partial object that has the class instance pre-filled as the first argument
        for the __call__ method.
    """

        return functools.partial(self.__call__, instance)
