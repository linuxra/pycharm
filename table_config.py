from utils.reports import Table, PerfTable
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

config = {
    "f1": {
        "class": "Table",
        "title": "Table for {date}",
        "filename": "Table_{date}.pdf"
    },
    "f2": {
        "class": "PerfTable",
        "title": "PerfTable for {date}",
        "filename": "PerfTable_{date}.pdf"
    }
}

def f1(start_date, end_date, counter):
    df = pd.DataFrame({"Function Name": ['f1'],
                       "Start Date": [start_date],
                       "End Date": [end_date],
                       "Counter": [counter]})
    return df


def f2(start_date, end_date, counter):
    df = pd.DataFrame({"Function Name": ['f2'],
                       "Start Date": [start_date],
                       "End Date": [end_date],
                       "Counter": [counter]})
    return df


def generate_dates(n_months):
    start_date = datetime.now()
    return [(start_date + relativedelta(months=i), start_date + relativedelta(months=i + 1) - relativedelta(days=1), i + 1)
            for i in range(n_months)]


def worker(func, dates):
    return func(*dates)


def parallel_execution1(func_list, dates_list, num):
    with ProcessPoolExecutor(max_workers=num) as executor:
        futures = {executor.submit(worker, func, dates): func.__name__ for func in func_list for dates in dates_list}
        dfs_dict = {func.__name__: [] for func in func_list}
        for future in as_completed(futures):
            func_name = futures[future]
            try:
                df = future.result()
                dfs_dict[func_name].append(df)
                print(f"Task {func_name} completed successfully.")
            except Exception as exc:
                print(f"Task {func_name} generated an exception: {exc}")
    return dfs_dict


def save_table(df, func_name, date):
    class_name = config[func_name]["class"]
    table_class = globals()[class_name]
    table_title = config[func_name]["title"].format(date=date.strftime('%Y-%m-%d'))
    filename = config[func_name]["filename"].format(date=date.strftime('%Y-%m-%d'))
    table = table_class(df, table_title)
    table.save(filename)


if __name__ == '__main__':
    functions = [f1, f2]
    dates = generate_dates(36)
    result_dfs = parallel_execution1(functions, dates, 6)

    for func_name, dfs in result_dfs.items():
        for df, date in zip(dfs, dates):
            print(f"Dataframe for function {func_name} for date {date}:")
            print(df)
            save_table(df, func_name, date[0])
