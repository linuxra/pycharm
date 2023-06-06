from logging_decorator_good1 import LoggingDecorator
from log_1 import logging_decorator


class Queries:
    def __init__(self, logger):
        self.logger = logger

    @logging_decorator(logger=None)
    def query_1(self, param1, param2):
        query = f"SELECT * FROM table1 WHERE column1 = {param1} AND column2 = {param2}"
        return query

    @logging_decorator(logger=None)
    def query_2(self, param1):
        query = f"SELECT * FROM table2 WHERE column1 = {param1}"
        return query
