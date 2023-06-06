## logging_decorator.py
# logging_decorator.py
import functools


class LoggingDecorator:
    """
    A class-based decorator for logging the call and return information of methods.
    """

    def __init__(self, func):
        """
        Initializes the decorator with the function to be decorated.

        Args:
            func (function): The function to be decorated.
        """
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, instance, *args, **kwargs):
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
        try:
            result = self.func(instance, *args, **kwargs)
        except Exception as e:
            if logger is not None:
                logger.error(f"Exception occurred while calling '{self.func.__name__}': {e}")
            raise
        else:
            if logger is not None:
                logger.info(f"Function '{self.func.__name__}' returned: {result}")
            return result

    def __get__(self, instance, owner):
        """
        Returns a functools.partial object that has the class instance pre-filled as
        the first argument for the __call__ method.

        Args:
            instance: The class instance on which the decorated method is accessed.
            owner: The owner class of the instance.

        Returns:
            A functools.partial object.
        """
        return functools.partial(self.__call__, instance)

