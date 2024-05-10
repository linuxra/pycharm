class ImmutableType(type):
    def __setattr__(cls, key, value):
        if key in cls.__dict__:
            raise AttributeError(f"{key} is immutable")
        super().__setattr__(key, value)


class Settings(metaclass=ImmutableType):
    BCOLOR = "black"
    HEADER_FACE_COLOR = "white"
    HEADER_TEXT_COLOR = "black"
    HEADER_FONT_SIZE = 16
    COLUMN_COLORS = ["#f2f2f2", "white"]
    CAT_COLORS = ["#f2f2f2", "white"]

import inspect

class MemoSettings:
    BG_COLOR = 'blue'
    MAX_SIZE = 1024

    def __init__(self):
        # Initialize variables in the caller's local scope
        self.inject_into_caller()

    def inject_into_caller(self):
        """Injects snake_case attributes into the caller's local scope based on class constants."""
        caller_frame = inspect.currentframe().f_back
        for attr_name in dir(self):
            if attr_name.isupper() and not attr_name.startswith('__'):
                attr_value = getattr(self, attr_name)
                caller_frame.f_locals[self.to_snake_case(attr_name)] = attr_value

    @staticmethod
    def to_snake_case(name):
        """Converts a given uppercase string to snake_case."""
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

# Usage in your script or Jupyter notebook cell
settings = MemoSettings()

# Now, bg_color and max_size are directly accessible without the settings prefix.
print(bg_color)  # Output: blue
print(max_size)  # Output: 1024

print(Settings.BCOLOR)


class SegmentMeta(type):
    def __new__(cls, name, bases, dct, **kwargs):
        segment = kwargs.get("segment")
        if segment == "finance":
            dct["pull_data"] = cls.finance_pull_data
        elif segment == "marketing":
            dct["pull_data"] = cls.marketing_pull_data
        else:
            raise ValueError(f"Unsupported segment: {segment}")
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def finance_pull_data(self):
        print("Pulling data for finance.")
        # Finance-specific data pull logic

    @staticmethod
    def marketing_pull_data(self):
        print("Pulling data for marketing.")
        # Marketing-specific data pull logic


def create_data_handler(segment):
    class_name = f"{segment.capitalize()}DataHandler"
    return SegmentMeta(class_name, (), {}, segment=segment)


# Usage example
FinanceDataHandler = create_data_handler("finance")
finance_handler = FinanceDataHandler()
finance_handler.pull_data()  # Outputs: Pulling data for finance.

MarketingDataHandler = create_data_handler("marketing")
marketing_handler = MarketingDataHandler()
marketing_handler.pull_data()  # Outputs: Pulling data for marketing.
