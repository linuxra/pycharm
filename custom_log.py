import logging
import os
from datetime import datetime
import pandas as pd
import functools


class CustomLogger:
    """
    A custom logger class that simplifies the process of configuring and using a logger.
    
    This class wraps the Python logging module and provides an interface to easily
    create, configure, and use a logger with both console and file output.
    """

    def __init__(self, name, log_level=logging.INFO, log_file=None, log_format=None):
        """
        Initialize a CustomLogger instance.

        :param name: The name of the logger.
        :param log_level: The logging level. Defaults to logging.INFO.
        :param log_file: The path to the log file. If provided, log messages will be written to the file.
        :param log_format: The format for log messages. Defaults to '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        if not log_format:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(log_format)

        if log_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file_with_timestamp = f"{log_file}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file_with_timestamp)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_log_level(self, log_level):
        """
        Set the logging level for this logger.

        :param log_level: The logging level.
        """
        self.logger.setLevel(log_level)

    def remove_handlers(self):
        """
        Remove all handlers from this logger.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def add_handler(self, handler):
        """
        Add a handler to this logger.

        :param handler: The handler to add.
        """
        self.logger.addHandler(handler)

    def info(self, message):
        """
        Log an info-level message.

        :param message: The message to log.
        """
        self.logger.info(message)

    def warning(self, message):
        """
        Log a warning-level message.

        :param message: The message to log.
        """
        self.logger.warning(message)

    def error(self, message):
        """
        Log an error-level message.

        :param message: The message to log.
        """
        self.logger.error(message)

    def debug(self, message):
        """
        Log a debug-level message.

        :param message: The message to log.
        """
        self.logger.debug(message)


# Create a logger instance with a log file
logger = CustomLogger('my_module', log_file='my_log_file')

# Change the log level to DEBUG
logger.set_log_level(logging.INFO)

# Log messages
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.debug('This is a debug message')

# Remove existing handlers and add a new file handler with a custom format
logger.remove_handlers()
custom_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
file_handler = logging.FileHandler('custom_format_log.log')
file_handler.setFormatter(logging.Formatter(custom_format))
logger.add_handler(file_handler)

# Log messages with the new format
logger.info('This is an info message with a custom format')
logger.warning('This is a warning message with a custom format')
logger.error('This is an error message with a custom format')
logger.debug('This is a debug message with a custom format')


def logging_decorator1(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                if isinstance(arg, pd.DataFrame):
                    return f"DataFrame(shape={arg.shape})"
                else:
                    return repr(arg)

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(
                    f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


def logging_decorator2(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(
                    f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


def logging_decorator3(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                if isinstance(arg, pd.DataFrame):
                    return f"DataFrame: \n{arg.head()}"
                elif isinstance(arg, str):
                    return arg
                else:
                    return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(
                    f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


def logging_decorator(logger):
    """
    A decorator to log information about the execution of decorated functions.
    
    This decorator logs the function call with its arguments and the returned result.
    It handles various argument types like DataFrame, string, int, float, list, tuple, and dict
    by providing informative and truncated string representations.

    :param logger: A logger object responsible for logging messages.
    :return: A decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                if isinstance(arg, pd.DataFrame):
                    return f"DataFrame: \n{arg.head()}"
                elif isinstance(arg, (str, int, float)):
                    return str(arg)
                elif isinstance(arg, (list, tuple)):
                    return f"{arg.__class__.__name__}({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
                elif isinstance(arg, dict):
                    return f"dict({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
                else:
                    return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(
                    f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}")
                raise

        return wrapper

    return decorator





@logging_decorator(logger)
def add(a, b):
    return a + b


result = add(10, 9)
logger.info(f"{result}")

import smtplib
from logging.handlers import RotatingFileHandler, SMTPHandler
from email.message import EmailMessage


class AdvancedCustomLogger(CustomLogger):
    """
    An advanced custom logger class that extends CustomLogger and adds more advanced features.
    """

    class LogLevelContext:
        """
        A context manager for temporarily changing the log level.
        """

        def __init__(self, logger, temporary_log_level):
            self.logger = logger
            self.original_log_level = logger.logger.level
            self.temporary_log_level = temporary_log_level

        def __enter__(self):
            self.logger.set_log_level(self.temporary_log_level)

        def __exit__(self, exc_type, exc_value, traceback):
            self.logger.set_log_level(self.original_log_level)

    def log_level_context(self, temporary_log_level):
        """
        Create a context manager for temporarily changing the log level.

        :param temporary_log_level: The temporary log level to use within the context.
        :return: A LogLevelContext instance.
        """
        return self.LogLevelContext(self, temporary_log_level)

    def add_email_notification(self, mailhost, fromaddr, toaddrs, subject, credentials=None, secure=None,
                               log_level=logging.ERROR):
        """
        Add email notifications for specific log levels.

        :param mailhost: The mail server host.
        :param fromaddr: The sender's email address.
        :param toaddrs: The recipients' email addresses (a list or tuple of strings).
        :param subject: The subject of the email.
        :param credentials: A tuple of the username and password to use for authentication (optional).
        :param secure: A tuple to enable a secure connection (optional).
        :param log_level: The log level for which email notifications should be sent. Defaults to logging.ERROR.
        """
        email_handler = SMTPHandler(mailhost, fromaddr, toaddrs, subject, credentials, secure)
        email_handler.setLevel(log_level)
        self.logger.addHandler(email_handler)

    def add_rotating_file_handler(self, log_file, max_bytes=10485760, backup_count=10, log_level=logging.INFO,
                                  log_format=None):
        """
        Add a rotating file handler to the logger.

        :param log_file: The path to the log file.
        :param max_bytes: The maximum file size (in bytes) before rotating. Defaults to 10MB.
        :param backup_count: The number of backup files to keep. Defaults to 10.
        :param log_level: The log level for the file handler. Defaults to logging.INFO.
        :param log_format: The format for log messages. Defaults to None, which uses the logger's default format.
        """
        if not log_format:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(log_format)
        rotating_file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        rotating_file_handler.setFormatter(formatter)
        rotating_file_handler.setLevel(log_level)
        self.logger.addHandler(rotating_file_handler)


from logging import Filter, Handler


class MoreAdvancedCustomLogger(AdvancedCustomLogger):
    """
    A more advanced custom logger class that extends AdvancedCustomLogger and adds even more advanced features.
    """

    class CustomLogFilter(Filter):
        """
        A custom log filter class to filter log records based on a user-defined condition.
        """

        def __init__(self, condition):
            super().__init__()
            self.condition = condition

        def filter(self, record):
            return self.condition(record)

    def add_log_filter(self, handler_name, condition):
        """
        Add a custom log filter to a specific handler based on a user-defined condition.

        :param handler_name: The name of the handler to add the filter to.
        :param condition: A callable that takes a log record as input and returns True if the log record should be processed, False otherwise.
        """
        custom_filter = self.CustomLogFilter(condition)
        for handler in self.logger.handlers:
            if handler.__class__.__name__ == handler_name:
                handler.addFilter(custom_filter)
                break

    def add_custom_handler(self, custom_handler: Handler):
        """
        Add a custom log handler to the logger.

        :param custom_handler: An instance of a custom log handler that inherits from the logging.Handler class.
        """
        self.logger.addHandler(custom_handler)


from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
from queue import Queue


class EvenMoreAdvancedCustomLogger(MoreAdvancedCustomLogger):
    """
    An even more advanced custom logger class that extends MoreAdvancedCustomLogger and adds further advanced features.
    """

    def add_timed_rotating_file_handler(self, log_file, when='D', interval=1, backup_count=30, log_level=logging.INFO,
                                        log_format=None):
        """
        Add a time-based rotating file handler to the logger.

        :param log_file: The path to the log file.
        :param when: The time unit for rotation. Defaults to 'D' for daily rotation.
        :param interval: The number of time units between rotations. Defaults to 1.
        :param backup_count: The number of backup files to keep. Defaults to 30.
        :param log_level: The log level for the file handler. Defaults to logging.INFO.
        :param log_format: The format for log messages. Defaults to None, which uses the logger's default format.
        """
        if not log_format:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(log_format)
        timed_rotating_file_handler = TimedRotatingFileHandler(log_file, when=when, interval=interval,
                                                               backupCount=backup_count)
        timed_rotating_file_handler.setFormatter(formatter)
        timed_rotating_file_handler.setLevel(log_level)
        self.logger.addHandler(timed_rotating_file_handler)

    def enable_async_logging(self, queue=None):
        """
        Enable asynchronous logging by wrapping all handlers in QueueHandlers and setting up a QueueListener.

        :param queue: An optional queue.Queue instance to use for async logging. If not provided, a new queue will be created.
        """
        if queue is None:
            queue = Queue()

        handlers = self.logger.handlers
        self.logger.handlers = []

        for handler in handlers:
            queue_handler = QueueHandler(queue)
            queue_handler.setLevel(handler.level)
            self.logger.addHandler(queue_handler)

        listener = QueueListener(queue, *handlers)
        listener.start()

        return listener


from logging.handlers import SocketHandler, MemoryHandler


class FurtherAdvancedCustomLogger(EvenMoreAdvancedCustomLogger):
    """
    A further advanced custom logger class that extends EvenMoreAdvancedCustomLogger and adds more advanced features.
    """

    def add_socket_handler(self, host, port, log_level=logging.INFO, log_format=None):
        """
        Add a SocketHandler for remote logging.

        :param host: The hostname of the remote log server.
        :param port: The port number of the remote log server.
        :param log_level: The log level for the SocketHandler. Defaults to logging.INFO.
        :param log_format: The format for log messages. Defaults to None, which uses the logger's default format.
        """
        if not log_format:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(log_format)
        socket_handler = SocketHandler(host, port)
        socket_handler.setFormatter(formatter)
        socket_handler.setLevel(log_level)
        self.logger.addHandler(socket_handler)

    def add_memory_handler(self, capacity, target_handler=None, log_level=logging.INFO, log_format=None):
        """
        Add a MemoryHandler for buffering log records.

        :param capacity: The maximum number of log records to buffer.
        :param target_handler: The target handler to which buffered log records should be sent. Defaults to None, which means the handler will not flush automatically.
        :param log_level: The log level for the MemoryHandler. Defaults to logging.INFO.
        :param log_format: The format for log messages. Defaults to None, which uses the logger's default format.
        """
        if not log_format:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(log_format)
        memory_handler = MemoryHandler(capacity, target=target_handler)
        memory_handler.setFormatter(formatter)
        memory_handler.setLevel(log_level)
        self.logger.addHandler(memory_handler)
