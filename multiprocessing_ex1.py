from multiprocessing import Process
from multiprocessing import Pool

def print_func(continent='Asia'):
    print('The name of continent is : ', continent)

def my_function(i):
    print(f'Function {i} is running')


if __name__ == "__main__":  # confirms that the code is under main function
    names = ['America', 'Europe', 'Africa']
    procs = []
    proc = Process(target=print_func)  # instantiating without any argument
    procs.append(proc)
    proc.start()

    # instantiating process with arguments
    for name in names:
        # print(name)
        proc = Process(target=print_func, args=(name,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()









    # list of arguments for your functions
    args = list(range(40))

    # create a pool with 20 processes
    with Pool(20) as p:
        # use map to apply my_function to each element in args
        p.map(my_function, args)


