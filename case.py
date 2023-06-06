from datetime import datetime, timedelta


def generate_monthly_case_statement(start_date):
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

def print_dict_rows(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}\n")

# Example usage

print_dict_rows(case_statements_dict)




def get_case_statement_for_month(case_statements_dict, num_month):
    return case_statements_dict.get(num_month)

# Example usage
num_month = 3
case_statement = get_case_statement_for_month(case_statements_dict, num_month)

print(case_statement)
