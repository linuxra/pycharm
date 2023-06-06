from typing import List, Optional
import pandas as pd
import threading
import datetime
import queue

class TeraUtils:
    # ...

    def execute_queries_transaction1(self, queries: List[str],
                                     return_data: bool = False,
                                     max_workers: int = 6) -> Optional[List[pd.DataFrame]]:
        """
        Execute multiple queries in a single transaction.

        :param queries: A list of SQL queries to execute.
        :param return_data: If True, return a list of pandas DataFrames containing the query results.
        :param max_workers: Maximum number of threads to run at a time.
        :return: A list of pandas DataFrames containing the query results, if return_data is True. None otherwise.
        """
        dataframes = []

        def execute_query(query: str) -> pd.DataFrame:
            start_time = datetime.datetime.now()
            print(f"Executing query: {query} at {start_time}")

            if return_data:
                df = pd.read_sql(query, self.connection)
            else:
                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    df = None

            end_time = datetime.datetime.now()
            duration = end_time - start_time
            print(f"Finished executing query: {query} at {end_time}. Duration: {duration}")
            return df

        def worker():
            while True:
                query = q.get()
                if query is None:
                    break

                if return_data:
                    df = execute_query(query)
                    dataframes.append(df)
                else:
                    execute_query(query)

                q.task_done()

        q = queue.Queue()
        threads = []
        for _ in range(max_workers):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        for query in queries:
            q.put(query)

        q.join()

        for _ in range(max_workers):
            q.put(None)

        for t in threads:
            t.join()

        try:
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"Error while executing queries in a transaction: {e}")

        return dataframes if return_data else None


from typing import List, Optional
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor

class TeraUtils:
    # ...

    def execute_queries_transaction1(self, queries: List[str],
                                     return_data: bool = False,
                                     max_workers: int = 6) -> Optional[List[pd.DataFrame]]:
        """
        Execute multiple queries in a single transaction.

        :param queries: A list of SQL queries to execute.
        :param return_data: If True, return a list of pandas DataFrames containing the query results.
        :param max_workers: Maximum number of threads to run at a time.
        :return: A list of pandas DataFrames containing the query results, if return_data is True. None otherwise.
        """
        dataframes = []

        def execute_query(query: str) -> pd.DataFrame:
            start_time = datetime.datetime.now()
            print(f"Executing query: {query} at {start_time}")

            if return_data:
                df = pd.read_sql(query, self.connection)
            else:
                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    df = None

            end_time = datetime.datetime.now()
            duration = end_time - start_time
            print(f"Finished executing query: {query} at {end_time}. Duration: {duration}")
            return df

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(execute_query, query) for query in queries]
                for future in futures:
                    if return_data:
                        dataframes.append(future.result())

            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"Error while executing queries in a transaction: {e}")

        return dataframes if return_data else None
