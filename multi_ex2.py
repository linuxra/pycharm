import logging
import multiprocessing

def worker(i):
    # Configure a logger for this process
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)

    # Create a log file handler for this process
    handler = logging.FileHandler('logfile.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Now you can use the logger as usual
    logger.info(f'Worker {i} - Starting')
    logger.info(f'Worker {i} - Exiting')

    # Remove the handler at the end
    logger.removeHandler(handler)
    handler.close()

if __name__ == '__main__':
    # Create some processes
    processes = []
    for i in range(10):
        process = multiprocessing.Process(target=worker, args=(i,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

