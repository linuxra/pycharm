from multiprocessing import Pool
import itertools
from datetime import date

# Your list of date tuples
date_tuples = [(date(2022, 1, 1), date(2022, 2, 1)), (date(2022, 3, 1), date(2022, 4, 1))]

# The functions you want to call with each date tuple
def func1(date_tuple):
    start, end = date_tuple
    return f'Function1 processed: {start} to {end}'

def func2(date_tuple):
    start, end = date_tuple
    return f'Function2 processed: {start} to {end}'

def func3(date_tuple):
    start, end = date_tuple
    return f'Function3 processed: {start} to {end}'

# Your list of functions
funcs = [func1, func2, func3]

# Create a pool of workers
with Pool() as p:
    # Call each function with each date tuple
    results = p.map(lambda args: args[0](args[1]), itertools.product(funcs, date_tuples))

print(results)
