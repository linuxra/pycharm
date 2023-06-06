# import time
#
# def timer_decorator(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         elapsed_time = time.time() - start_time
#         print(f"Elapsed time: {elapsed_time:.2f} seconds")
#         return result
#     return wrapper
#
# @timer_decorator
# def some_operation():
#     time.sleep(2)
#
# some_operation()
#
# import time
#
# class Timer:
#     def __enter__(self):
#         self.start_time = time.time()
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         elapsed_time = time.time() - self.start_time
#         print(f"Elapsed time: {elapsed_time:.2f} seconds")
#
# with Timer():
#     # Perform some operation
#     time.sleep(2)
#
# import cProfile

# def profile(func):
#     def wrapper(*args, **kwargs):
#         profiler = cProfile.Profile()
#         result = profiler.runcall(func, *args, **kwargs)
#         profiler.print_stats()
#         return result
#     return wrapper

# @profile
# @timer_decorator
# def some_operation():
#     # Perform some operation
#     for i in range(1000000):
#         pass
#
# some_operation()

