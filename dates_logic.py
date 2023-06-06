from datetime import datetime
from dateutil.relativedelta import relativedelta

# Start date
start_date = datetime.strptime('2021-12-01', '%Y-%m-%d')

# Current date
current_date = datetime.now()

# Start from the next month
start_date = start_date + relativedelta(months=1)

# Generate list of dates
dates = []
for i in range(24):
    new_date = start_date + relativedelta(months=i)
    if new_date.year > current_date.year or (new_date.year == current_date.year and new_date.month >= current_date.month):
        break
    dates.append(new_date.strftime('%y%m'))
print(dates)

# Dictionary of fruits and their colors
fruits = {'apple': 'red', 'banana': 'yellow', 'mango': 'orange'}

for i, fruit in enumerate(fruits):
    print(f"fruit number {i} is {fruit} and its color is {fruits[fruit]}")
