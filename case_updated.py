from typing import Dict
from datetime import datetime, timedelta


def generate_monthly_case_statement(start_date: str) -> Dict[int, str]:
    """
    Generates a dictionary of SQL case statements for each month up to the current month, limited to the past 24 months.

    :param str_date: The starting date for generating case statements, in the format 'YYYYMM'.
    :return: A dictionary with keys being the month number and values being the corresponding SQL case statements.
    """
    current_date = datetime.now()
    start_date = datetime.strptime(start_date, "%Y%m")

    num_months = (current_date.year - start_date.year) * 12 + current_date.month - start_date.month - 1
    num_months = min(num_months, 24)

    base_condition = "substring(pay_code, {}, 1) in (1, 2, 3, 4)"
    conditions = [base_condition.format(13)]

    for i in range(1, 12):
        conditions.append(base_condition.format(13 - i))

    case_statements = {}

    for i, condition in enumerate(conditions[:num_months]):
        month_conditions = " OR ".join(conditions[:i + 1])
        case_statements[i + 1] = f"WHEN {month_conditions} THEN 1 ELSE 0 END AS bad"

    if num_months > 12:
        for i in range(12, num_months):
            case_statements[i + 1] = case_statements[12]

    return case_statements


# Example usage
start_date = "202004"
case_statements_dict = generate_monthly_case_statement(start_date)


def print_dict_rows(dictionary: Dict[int, str]) -> None:
    """
    Prints the key-value pairs of a dictionary in a formatted way.

    :param dictionary: The dictionary to print.
    """
    for key, value in dictionary.items():
        print(f"{key}: {value}\n")


# Example usage

print_dict_rows(case_statements_dict)


def get_case_statement_for_month(case_statements_dict: Dict[int, str], num_month: int) -> str:
    """
    Retrieves the case statement for the specified month from the provided case statements dictionary.

    :param case_statements_dict: A dictionary containing case statements.
    :param num_month: The month number to retrieve the case statement for.
    :return: The case statement for the specified month.
    """
    return case_statements_dict.get(num_month)


# Example usage
num_month = 3
case_statement = get_case_statement_for_month(case_statements_dict, num_month)

print(case_statement)
