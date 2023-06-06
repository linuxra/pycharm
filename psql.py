def generate_table_name(table_prefix, month_offset):
    return f"{table_prefix}_{month_offset:02}"

def generate_sql_query(month, table_prefix='table_name'):
    month_subqueries = "\n".join(
        f"SELECT '{month + i}' AS month, acc_id, prod_id, ... FROM {generate_table_name(table_prefix, month + i)}"
        for i in range(1, 25)
    )

    return f"""
    WITH account_selection AS (
      SELECT
        t1.acc_id,
        t1.prod_id,
        t1.fico_score,
        CASE
          WHEN t1.fico_score >= 600 AND t1.fico_score < 700 THEN 600
          WHEN t1.fico_score >= 700 AND t1.fico_score < 800 THEN 700
          -- ...
          ELSE NULL
        END AS score_range,
        CASE
          WHEN t1.fico_score >= 600 AND t1.fico_score < 700 THEN 1
          WHEN t1.fico_score >= 700 AND t1.fico_score < 800 THEN 2
          -- ...
          ELSE NULL
        END AS rank
      FROM
        table_1 AS t1
        LEFT JOIN table_2 AS t2 ON t1.acc_id = t2.acc_id
        LEFT JOIN table_3 AS t3 ON t1.acc_id = t3.acc_id
      WHERE
        -- Add your conditions to select the accounts
    ),
    bad_accounts AS (
      SELECT
        acc_id,
        prod_id,
        score_range,
        rank,
        month,
        SUM(
          CASE
            -- Replace "bad_condition" with the actual condition for bad accounts
            WHEN bad_condition THEN 1
            ELSE 0
          END
        ) AS bad_count
      FROM
        account_selection
        LEFT JOIN (
          {month_subqueries}
        ) AS monthly_data ON account_selection.acc_id = monthly_data.acc_id
      GROUP BY
        acc_id, prod_id, score_range, rank, month
    ),
    pivot_data AS (
      SELECT
        rank,
        score_range,
        month - {month} AS t,
        ROW_NUMBER() OVER (PARTITION BY rank, score_range ORDER BY month) AS counter,
        SUM(bad_count) AS bad_count
      FROM
        bad_accounts
      GROUP BY
        rank, score_range, month
    )
    SELECT * FROM pivot_data
    ORDER BY rank, score_range, t;
    """
month = 2201  # Replace this with the actual month value
table_prefix = 'table_name'  # Replace this with the actual table prefix
sql_query = generate_sql_query(month, table_prefix)
print(sql_query)