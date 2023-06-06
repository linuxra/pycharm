from datetime import datetime, timedelta

class DateUtils:
    @staticmethod
    def generate_month_ranges(yyyymm: str, n: int):
        """
        Generate a list of tuples containing the start and end dates of previous months,
        excluding the input month.

        :param yyyymm: A string representing the month in YYYYMM format.
        :param n: The number of previous months to generate.
        :return: A list of tuples containing the start and end dates of the previous months
                 in 'YYYYMMDD' format.
        """
        def month_range(date):
            start_date = date.replace(day=1)
            end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            return start_date, end_date

        current_month = datetime.strptime(yyyymm, "%Y%m")
        month_ranges = []
        for i in range(1, n + 1):
            prev_month = current_month - timedelta(days=i * 28)  # Approximate number of days in a month
            start_date, end_date = month_range(prev_month)
            month_ranges.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))

        return month_ranges

    @staticmethod
    def generate_next_yymms(date_str: str, n: int):
        """
        Generate a list of YYMM strings for the next N months, excluding the given date's month
        and the current month. Stop generating months if N crosses the current month.

        :param date_str: A string representing the date in YYYYMMDD format.
        :param n: The number of next months to generate.
        :return: A list of YYMM strings for the next N months.
        """
        date = datetime.strptime(date_str, "%Y%m%d")
        current_month = datetime.now().replace(day=1)
        yymms = []

        for i in range(1, n + 1):
            next_month = (date.replace(day=1) + timedelta(days=i * 30)).replace(
                day=1)  # Approximate number of days in a month
            if next_month >= current_month:
                break

            yymm = next_month.strftime("%y%m")
            yymms.append(yymm)

        return yymms


# Example usage
month_ranges = DateUtils.generate_month_ranges("202212", 12)
for start_date, end_date in month_ranges:
    print(start_date, end_date)

    # Example usage
next_yymms = DateUtils.generate_next_yymms("20230101", 15)
print(next_yymms)
