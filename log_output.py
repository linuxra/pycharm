from functools import wraps
import sys
from io import StringIO


def log_output_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect stdout to a StringIO object
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = func(*args, **kwargs)
            log_message = sys.stdout.getvalue().strip()
            if log_message:
                logger.info(log_message)
        finally:
            # Restore the original stdout
            sys.stdout = old_stdout

        return result

    return wrapper
