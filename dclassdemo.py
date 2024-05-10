import sys


class ImmutableType(type):
    """
    A metaclass that makes attributes of the class immutable once they are set and
    injects these attributes into the global namespace of the module as lowercase variables.
    Attempts to modify existing class attributes will result in an AttributeError.

    This metaclass also includes a __new__ method for initializing the class and automatically
    setting class attributes as global variables in lowercase form.
    """

    def __new__(cls, name, bases, dct):
        # Create the class normally
        new_class = super().__new__(cls, name, bases, dct)
        # Inject class attributes into the global namespace of the module where the class is defined
        module = sys.modules[new_class.__module__]
        for attr_name in dct:
            if not attr_name.startswith("__"):
                # Set the attribute in the global namespace of the module as lowercase
                setattr(module, attr_name.lower(), dct[attr_name])
        return new_class

    def __setattr__(cls, key, value):
        if key in cls.__dict__:
            raise AttributeError(f"{key} is immutable")
        super().__setattr__(key, value)


class Settings(metaclass=ImmutableType):
    """
    A settings class where attributes are defined as constants and cannot be modified
    once the class is created. This class is useful for defining application settings
    that should not change during the application's lifetime.

    Attributes:
        BCOLOR (str): Background color, default is 'black'.
        HEADER_FACE_COLOR (str): Face color for headers, default is 'white'.
        HEADER_TEXT_COLOR (str): Text color for headers, default is 'black'.
        HEADER_FONT_SIZE (int): Font size for headers, default is 16.
        COLUMN_COLORS (list): Default colors for columns, list contains shades of gray and white.
        CAT_COLORS (list): Default category colors, list contains shades of gray and white.
    """


# Accessing the global variables directly
print(bcolor)  # Output: black
print(header_face_color)  # Output: white

# Example of trying to modify an immutable attribute
try:
    Settings.BCOLOR = "red"
except AttributeError as e:
    print(e)  # Output: BCOLOR is immutable

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
