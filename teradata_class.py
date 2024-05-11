import teradatasql
import pandas as pd
import logging


class TeradataConnection:
    """
    Context manager for managing Teradata database connections with extended functionality for executing queries and logging.

    Attributes:
        host (str): Hostname of the Teradata database server.
        username (str): Username for database login.
        password (str): Password for database login.
        database (str): Database name to connect to.
        logger (logging.Logger): Logger object for logging status and errors.
    """

    def __init__(self, host, username, password, database, logger):
        """
        Initialize the TeradataConnection context manager.
        """
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.logger = logger
        self.connection = None

    def __enter__(self):
        """
        Establish a connection to the Teradata database.
        """
        try:
            self.connection = teradatasql.connect(
                host=self.host,
                user=self.username,
                password=self.password,
                database=self.database,
            )
            self.logger.info("Database connection established.")
            return self
        except Exception as e:
            self.logger.error("Failed to connect to the database: %s", str(e))
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Close the connection to the Teradata database.
        """
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed.")

    def execute_query(self, sql):
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        """
        try:
            df = pd.read_sql(sql, self.connection)
            self.logger.info(
                f"Query executed successfully. Returned {len(df)} records."
            )
            return df
        except Exception as e:
            self.logger.error("Failed to execute query: %s", str(e))
            raise

    def execute_query_conditional(self, sql, create_df=True):
        """
        Execute a SQL query and optionally return the results as a pandas DataFrame or cursor.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            if create_df:
                df = pd.DataFrame(
                    cursor.fetchall(), columns=[col[0] for col in cursor.description]
                )
                self.logger.info(
                    f"Query executed successfully. Returned {len(df)} records as DataFrame."
                )
                return df
            else:
                self.logger.info("Query executed successfully. Returning cursor.")
                return cursor
        except Exception as e:
            self.logger.error("Failed to execute query: %s", str(e))
            raise

    def get_cursor(self):
        """
        Get a cursor object from the current database connection.
        """
        try:
            cursor = self.connection.cursor()
            self.logger.info("Cursor obtained successfully.")
            return cursor
        except Exception as e:
            self.logger.error("Failed to obtain cursor: %s", str(e))
            raise

    def execute_query1(self, sql, as_dataframe=True):
        """
        Execute a SQL query and optionally return results as a pandas DataFrame.

        Args:
            sql (str): The SQL query to execute.
            as_dataframe (bool): If True (default), returns a DataFrame.
                                 If False, returns the cursor object for direct iteration.

        Returns:
            pd.DataFrame or teradatasql.Cursor: The query result or the cursor.
        """
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(sql)
                if as_dataframe:
                    df = pd.DataFrame(
                        cursor.fetchall(),
                        columns=[col[0] for col in cursor.description],
                    )
                    self.logger.info(
                        f"Query executed successfully. Returned {len(df)} rows."
                    )
                    return df
                else:
                    self.logger.info("Query executed successfully. Returning cursor.")
                    return cursor
            except teradatasql.Error as e:
                self.logger.error(f"Failed to execute query: {e}")
                raise

    def execute_query2(self, sql, as_dataframe=True, batch_size=1000):
        """
        Execute a SQL query and optionally return results as a pandas DataFrame in batches.

        Args:
            sql (str): The SQL query to execute.
            as_dataframe (bool): If True (default), yields DataFrames in batches.
                                 If False, yields the cursor object for direct iteration.
            batch_size (int): The number of rows to fetch per batch (default: 1000).

        Yields:
            pd.DataFrame or teradatasql.Cursor: The query result in batches or the cursor.
        """
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(sql)
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break  # No more rows to fetch
                    if as_dataframe:
                        yield pd.DataFrame(
                            rows, columns=[col[0] for col in cursor.description]
                        )
                    else:
                        yield rows
                self.logger.info("Query executed successfully.")
            except teradatasql.Error as e:
                self.logger.error(f"Failed to execute query: {e}")
                raise

    import pandas as pd
    import teradatasql

    def execute_query3(self, sql, as_dataframe=True, batch_size=1000):
        """
        Execute a SQL query and optionally return results as a pandas DataFrame in batches.
        This method leverages the power of Python's generators to manage large datasets efficiently.

        Args:
            sql (str): The SQL query to execute.
            as_dataframe (bool): If True (default), yields DataFrames in batches.
                                 If False, yields the cursor object for direct iteration.
            batch_size (int): The number of rows to fetch per batch (default: 1000).

        Yields:
            pd.DataFrame or list: If `as_dataframe` is True, yields a DataFrame for each batch.
                                  If `as_dataframe` is False, yields lists of tuples representing rows.

        Raises:
            RuntimeError: If there is no active database connection.
            Exception: Propagates any database-specific errors that occur during query execution.
        """
        if not self.connection:
            self.logger.error("Attempted to execute a query without an active database connection.")
            raise RuntimeError("No active database connection.")

        try:
            with self.connection.cursor() as cursor:
                self.logger.info(f"Executing SQL query: {sql}")
                cursor.execute(sql)
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        self.logger.info("No more rows to fetch.")
                        break  # No more rows to fetch
                    if as_dataframe:
                        df = pd.DataFrame(rows, columns=[col[0] for col in cursor.description])
                        self.logger.info(f"Yielding a DataFrame with {len(df)} rows.")
                        yield df
                    else:
                        self.logger.info(f"Yielding raw data rows.")
                        yield rows
        except teradatasql.Error as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise
        finally:
            self.logger.info("Query execution completed.")
