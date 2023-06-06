# another_class.py
from log_1 import logging_decorator


class AnotherClass:
    def __init__(self, logger):
        self.logger = logger

    @logging_decorator(logger=None)
    def another_method(self, param1):
        result = f"Result: {param1}"
        return result
