from datetime import datetime, timedelta

def get_date_info(yyyymm, window):
    # Convert yyyymm to datetime object
    current_date = datetime.strptime(yyyymm, "%Y%m")

    # Calculate score_date by subtracting window months
    month = current_date.month - window % 12
    year = current_date.year - window // 12
    if month <= 0:
        month += 12
        year -= 1
    score_date = datetime(year, month, 1)

    # Find start_date and end_date of the score_date month
    start_date = score_date
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    # Format the given yyyymm date as perfyymm (yyMM format)
    perfyymm = current_date.strftime("%y%m")

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), score_date.strftime("%Y%m"), perfyymm

# Example usage
yyyymm = "202012"
window = 10
result = get_date_info(yyyymm, window)
print(result)

from datetime import datetime, timedelta

def generate_monthly_dates(start_date):
    # Parse the start date
    start = datetime.strptime(start_date, "%Y%m")

    # Get the current date
    now = datetime.now()

    # Generate a list to hold the dates
    dates = []

    # Loop from the start date to the current date
    current = start
    while current.year < now.year or (current.year == now.year and current.month <= now.month):
        # Append the date in YYYYMM format
        dates.append(current.strftime("%Y%m"))
        # Move to the next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return dates

# Example usage
start_date = "202201"
dates = generate_monthly_dates(start_date)
print(dates)
