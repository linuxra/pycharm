from datetime import datetime


def last_quarter(date_str):
    # Parse the date string into a datetime object
    current_date = datetime.strptime(date_str, '%Y-%m-%d')

    year = current_date.year
    month = current_date.month
    quarter = (month - 1) // 3 + 1

    last_quarter = quarter - 1
    if last_quarter == 0:
        last_quarter = 4
        year -= 1

    return f'{year}Q{last_quarter}'


# test the function
print(last_quarter('2023-01-28'))

from datetime import datetime
from dateutil.relativedelta import relativedelta


def last_quarter_and_36_months_ago(date_str):
    # Parse the date string into a datetime object
    current_date = datetime.strptime(date_str, '%Y-%m-%d')

    # Subtract 2 months from the current date
    adjusted_date = current_date - relativedelta(months=2)

    year = adjusted_date.year
    month = adjusted_date.month

    # Subtract 36 months
    month_36_ago = adjusted_date - relativedelta(months=35)

    # Convert dates to the required format
    last_end_month_str = adjusted_date.strftime('%Y%m')
    month_36_ago_str = month_36_ago.strftime('%Y%m')

    return month_36_ago_str , last_end_month_str

# Test the function
print(last_quarter_and_36_months_ago('2023-06-28'))

from datetime import datetime
from dateutil.relativedelta import relativedelta

def months_between(date1_str, date2_str):
    # Parse the date strings into datetime objects
    date1 = datetime.strptime(date1_str, '%Y%m')
    date2 = datetime.strptime(date2_str, '%Y%m')

    # Compute the difference between the two dates
    difference = relativedelta(date2, date1)

    # Return the number of months
    return difference.years * 12 + difference.months + 1  # add 1 to include both months

# Test the function
print(months_between('202005', '202304'))


