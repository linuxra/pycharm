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
