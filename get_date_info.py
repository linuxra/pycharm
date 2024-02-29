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

