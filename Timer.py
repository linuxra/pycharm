import time
import logging


class Timer:
    """
    A timer utility class that can be used as a context manager or via direct method calls.
    This class provides a simple way to measure the execution time of code blocks or functions,
    and supports logging the times via a provided logger object.

    Attributes:
        start_time (float): The start time of the timer.
        end_time (float): The end time of the timer.
        elapsed (float): The elapsed time in seconds.
        logger (logging.Logger): Logger object for logging time measurements.

    Examples:
        Using as a context manager with logging:
            logger = logging.getLogger(__name__)
            with Timer(logger) as t:
                # some code to time
            print(f"Elapsed time: {t.elapsed} seconds")

        Using via method calls with logging:
            logger = logging.getLogger(__name__)
            t = Timer(logger)
            t.start()
            # some code to time
            t.stop()
            print(f"Elapsed time: {t.elapsed} seconds")
    """

    def __init__(self, logger=None):
        """
        Initializes the Timer instance, setting all times to None and configuring the logger.

        Parameters:
            logger (logging.Logger): Optional logger for logging start, end, and elapsed times.
        """
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug("Timer initialized without starting.")

    def start(self):
        """
        Start the timer by recording the current time. This method sets the start_time and logs the event.
        """
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None
        self.logger.debug(f"Timer started at {self.start_time}.")

    def stop(self):
        """
        Stop the timer by recording the current time and calculating the elapsed time.
        This method sets the end_time and updates the elapsed attribute to show the time difference
        between start_time and end_time. It also logs the stop event and elapsed time.
        """
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.logger.debug(
            f"Timer stopped at {self.end_time}. Elapsed time: {self.elapsed} seconds."
        )

    def __enter__(self):
        """
        Enter the runtime context related to this object. The with statement will bind this method’s
        return value to the target specified in the as clause of the statement, if any.
        In this case, it starts the timer and logs the start event.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and stop the timer. This method is called upon exiting the with block.
        It ensures that the timer stops and calculates the elapsed time, logs the stop event, and handles
        any exception that occurred.

        Parameters:
            exc_type: Exception type, if raised in the with block.
            exc_val: Exception value, if raised.
            exc_tb: Traceback object, if an exception was raised.
        """
        self.stop()
        # Can optionally handle exceptions here or log details
        # Return False to propagate exceptions, True to suppress them
        return False
