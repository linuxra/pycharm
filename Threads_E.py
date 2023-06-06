import threading
import pandas as pd

# Define your function that runs multiple SQL queries and returns a DataFrame
def my_function(param1, param2):
    # SQL query 1
    # SQL query 2
    # ...
    # SQL query n
    df = pd.DataFrame(...)  # Create a DataFrame from the query results
    return df

# Define a list of parameters to pass to the function
params_list = [(1, 'a'), (2, 'b'), (3, 'c')]

# Define a list to hold the results of each function call
results_list = []

# Define a function that runs my_function with a given set of parameters and stores the result
def run_function(params):
    result = my_function(*params)
    results_list.append(result)

# Create a thread for each set of parameters and start the threads
threads = [threading.Thread(target=run_function, args=(params,)) for params in params_list]
for thread in threads:
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Combine the results into a single DataFrame
final_df = pd.concat(results_list)
