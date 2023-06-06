# logging_decorator.py
# class LoggingDecorator:
#     logger = None
#
#     @classmethod
#     def set_logger(cls, logger):
#         cls.logger = logger
#
#     def __init__(self, func):
#         self.func = func
#
#     def __call__(self, *args, **kwargs):
#         if LoggingDecorator.logger is not None:
#             LoggingDecorator.logger.info(f"Calling function '{self.func.__name__}' with args: {args} and kwargs: {kwargs}")
#         result = self.func(*args, **kwargs)
#         if LoggingDecorator.logger is not None:
#             LoggingDecorator.logger.info(f"Function '{self.func.__name__}' returned: {result}")
#         return result
from logging_decorator import LoggingDecorator
import logging

# queries.py


# queries.py
# queries.py
class Queries:
    def __init__(self, logger):
        self.logger = logger

    @LoggingDecorator
    def query_1(self, param1, param2):
        query = f"SELECT * FROM table1 WHERE column1 = {param1} AND column2 = {param2}"
        return query

    @LoggingDecorator
    def query_2(self, param1):
        query = f"SELECT * FROM table2 WHERE column1 = {param1}"
        return query



# Set up the logger


# Set up the logger


# Set up the logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create a Queries object with the logger
queries = Queries(logger)

# Call the decorated methods
query1_result = queries.query_1("John", "Doe")
query2_result = queries.query_2(42)


