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


def display_macro(file_path, styles1, COLUMN_COLOR, COL_TXT_COLOR):
    """
    Displays a macro from a specified file, applying DataFrame styling and uniform color customization
    to column backgrounds and text to enhance visual presentation. The macro data is assumed to be stored
    in a format readable by pandas, such as CSV or Excel. The method reads the file, applies the provided
    DataFrame styles, and sets specified colors for all column backgrounds and text.

    Parameters:
        file_path (str): The path to the file containing the macro data. This should be a path to a
                         readable file format by pandas, such as '.csv' or '.xlsx'.
        styles1 (pandas.io.formats.style.Styler): A pandas Styler object containing styles to be
                                                  applied to the DataFrame. This object configures
                                                  how the DataFrame should be displayed, such as
                                                  background color, text alignment, and other
                                                  visual styles.
        COLUMN_COLOR (str): A color string (e.g., 'yellow', 'red') used to set the background color
                            for all columns in the DataFrame.
        COL_TXT_COLOR (str): A color string (e.g., 'black', 'white') used to set the text color for
                             all columns in the DataFrame.

    Returns:
        None. The method is used for its side effect of displaying a styled DataFrame within
        a Jupyter Notebook or similar environment.

    Example:
        # Assuming 'data.csv' contains columns that need to be styled
        file_path = 'data.csv'
        styles1 = df.style.applymap(lambda x: 'background-color: lightgrey')
        COLUMN_COLOR = 'yellow'
        COL_TXT_COLOR = 'black'
        display_macro(file_path, styles1, COLUMN_COLOR, COL_TXT_COLOR)
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    styled_df = df.style.apply(styles1)  # Apply general styles
    # Apply uniform column and text colors
    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th, td",
                "props": [("background-color", COLUMN_COLOR), ("color", COL_TXT_COLOR)],
            }
        ]
    )
    display(styled_df)  # display is typically from IPython.display or similar
