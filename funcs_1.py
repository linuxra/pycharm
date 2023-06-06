@manage_db_connection(logger)
@logging_decorator(logger)
def execute_sql_query(cursor, sql_string):
    """
    Execute the SQL query and fetch the result.

    Args:
        cursor: Database cursor object.
        sql_string: SQL query string.

    Returns:
        The result of the query.

    """
    try:
        with cursor as cursor:
            # Execute the SQL query
            cursor.execute(sql_string)

            # Fetch the result
            result = cursor.fetchone()[0] or 0
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        result = 0

    return result


def rail_acq_new(aprtera, cursor, yymm, start_date, end_date, thread_id, qtr):
    """
    Retrieve the count of new accounts for rail acquisition.

    Args:
        aprtera: APRTERA object.
        cursor: Database cursor object.
        yymm: Year and month parameter.
        start_date: Start date for the query.
        end_date: End date for the query.
        thread_id: Thread ID parameter.
        qtr: Quarter parameter.

    Returns:
        The count of new accounts.

    """
    # Generate the SQL query string
    sql_string = queries.new_accts_rail(yymm, start_date, end_date)

    # Execute the SQL query and fetch the result
    count = execute_sql_query(cursor, sql_string)

    return count


def rail_CREDIT_LINE1(aprtera, cursor, yymm, start_date, end_date, thread_id, qtr):
    """
    Retrieve the count of new accounts for rail acquisition.

    Args:
        aprtera: APRTERA object.
        cursor: Database cursor object.
        yymm: Year and month parameter.
        start_date: Start date for the query.
        end_date: End date for the query.
        thread_id: Thread ID parameter.
        qtr: Quarter parameter.

    Returns:
        The count of new accounts.

    """
    # Generate the SQL query string
    sql_string = queries.rai1_sql_credit_line(yymm, start_date, end_date)

    # Execute the SQL query and fetch the result
    count = execute_sql_query(cursor, sql_string)

    return count
