import logging

import logging
from queries import Queries
from another_class import AnotherClass
from func_file import square, M
from log_1 import logging_decorator


from log_1 import logging_decorator
# Set up the logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
result = logging_decorator(logger)(square)
result1 = logging_decorator(logger)(M.add)
result1(3,6)
print(result(6))
# Create a Queries object with the logger
queries = Queries(logger)

@logging_decorator(logger)
def simple(x,y):
    """

    :type y: object
    """
    return x + y
simple(4,5)
# Call the decorated methods of Queries
query1_result = queries.query_1("John", "Doe")
query2_result = queries.query_2(42)



# Create an AnotherClass object with the logger
another_instance = AnotherClass(logger)

# Call the decorated method of AnotherClass
another_method_result = another_instance.another_method("example")
