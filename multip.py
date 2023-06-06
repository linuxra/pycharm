import concurrent.futures
import multiprocessing_ex1
import random

def process_data(shared_array, start_index, end_index):
    for i in range(start_index, end_index):
        shared_array[i] = shared_array[i] * 2

def main():
    data_length = 1000000
    chunk_size = 10000

    # Generate some random data
    data = [random.randint(1, 100) for _ in range(data_length)]

    # Create a shared array using multiprocessing.Array
    shared_array = multiprocessing.Array('i', data)

    # Define the chunks of the dataset to be processed in parallel
    chunk_indices = [(i * chunk_size, (i + 1) * chunk_size) for i in range(data_length // chunk_size)]

    # Process the data in parallel using concurrent.futures.ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_data, [shared_array] * len(chunk_indices), *zip(*chunk_indices))

    # Print the first 10 elements of the processed data
    print(shared_array[:10])

if __name__ == '__main__':
    main()
