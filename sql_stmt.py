## Define the start and end dates for each month
dates = [
    ('2023-01-01', '2023-01-31', '2301'),
    ('2023-02-01', '2023-02-28', '2302'),
    ('2023-03-01', '2023-03-31', '2303'),
    ('2023-04-01', '2023-04-30', '2304'),
    ('2023-05-01', '2023-05-31', '2305'),
    ('2023-06-01', '2023-06-30', '2306'),
    ('2023-07-01', '2023-07-31', '2307'),
    ('2023-08-01', '2023-08-31', '2308'),
    ('2023-09-01', '2023-09-30', '2309'),
    ('2023-10-01', '2023-10-31', '2310'),
    ('2023-11-01', '2023-11-30', '2311'),
    ('2023-12-01', '2023-12-31', '2312'),
]

# Define the base tables
tables = ['table1', 'table11', 'table111']

# Create a list of SQL statements to be combined with UNION ALL
sql_statements = []
for start_date, end_date, yymm in dates:
    for i, table in enumerate(tables):
        table_yymm = table + "_" + yymm + " AS " + chr(97 + i)  # chr(97 + i) will generate 'a', 'b', 'c'...
        sql_statements.append("SELECT col1, col2 FROM {} WHERE date_col BETWEEN '{}' AND '{}'".format(table_yymm, start_date, end_date))

# Build the SQL query string with UNION ALL
union_query = " UNION ALL ".join(sql_statements)

# Define the id value
id_value = 900

# Combine the CTE and final SELECT statement
final_query = """
WITH cte AS (
    {}
)
SELECT *
FROM cte
WHERE id = {}
""".format(union_query, id_value)

# Print the final SQL query
print(final_query)
