import calendar
import datetime


def get_month_bounds(date):
    # Get the year and month from the given date
    year = date.year
    month = date.month

    # Get the number of days in the month and the weekday of the first day
    _, num_days = calendar.monthrange(year, month)

    # Calculate the beginning and end dates of the month
    start_date = datetime.date(year, month, 1)
    end_date = datetime.date(year, month, num_days)

    return start_date, end_date


# Example usage
date = datetime.date(2021, 12, 31)
start_date, end_date = get_month_bounds(date)
print(f"Start date: {start_date}")
print(f"End date: {end_date}")
