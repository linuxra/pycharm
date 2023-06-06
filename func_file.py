from log_output import log_output_decorator



def square(x):
    return x ** 2


class M:

    @staticmethod
    def add(x,y):
        print(x + y)
