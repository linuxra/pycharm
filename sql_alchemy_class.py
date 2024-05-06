import pandas as pd

from sqlalchemy import text, inspect, and_, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from models import NetflixMovie, Base
from database import Database
from sqlalchemy import and_
from models import NetflixMovie, AnotherTable, ThirdTable, Base
from database import Database


class QueryManager:
    """
    A class used to manage database queries and operations using SQLAlchemy.

    Methods
    -------
    list_tables()
        Returns a list of all table names in the database.

    get_netflix_movies(limit=5)
        Fetches a specified number of Netflix movies from the database.

    introspect_table(table_name)
        Returns the column information for a specified table.

    execute_raw_query(query)
        Executes a raw SQL query and returns the result.

    upload_dataframe(df, table_name, if_exists='replace', index=False)
        Uploads a pandas DataFrame to a specified table.

    query_to_dataframe(query)
        Executes a SQL query and returns the result as a pandas DataFrame.

    left_join_two_tables(table1, table2, table1_id, table2_id, select_columns=None, where_conditions=None)
        Performs a left join between two tables with optional select columns and where conditions.

    left_join_three_tables(table1, table2, table3, table1_id, table2_id, table3_id, select_columns=None, where_conditions=None)
        Performs a left join between three tables with optional select columns and where conditions.
    """

    def __init__(self):
        """
        Initializes the QueryManager with a SQLAlchemy engine and session.
        """
        self.db = Database()
        self.engine = self.db.get_engine()
        self.session = self.db.get_session()

    def list_tables(self):
        """
        Returns a list of all table names in the database.

        Returns
        -------
        list
            A list of table names.
        """
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_netflix_movies(self, limit=5):
        """
        Fetches a specified number of Netflix movies from the database.

        Parameters
        ----------
        limit : int, optional
            The number of movies to fetch (default is 5).

        Returns
        -------
        list
            A list of NetflixMovie objects.
        """
        return self.session.query(NetflixMovie).limit(limit).all()

    def introspect_table(self, table_name):
        """
        Returns the column information for a specified table.

        Parameters
        ----------
        table_name : str
            The name of the table to introspect.

        Returns
        -------
        list
            A list of column information dictionaries.
        """
        inspector = inspect(self.engine)
        return inspector.get_columns(table_name)

    def execute_raw_query(self, query):
        """
        Executes a raw SQL query and returns the result.

        Parameters
        ----------
        query : str
            The raw SQL query to execute.

        Returns
        -------
        list
            A list of rows returned by the query.
        """
        with self.engine.connect() as connection:
            return connection.execute(text(query))

    def upload_dataframe(self, df, table_name, if_exists="replace", index=False):
        """
        Uploads a pandas DataFrame to a specified table.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to upload.
        table_name : str
            The name of the table to upload to.
        if_exists : str, optional
            What to do if the table exists (default is 'replace').
        index : bool, optional
            Whether to write the DataFrame index as a column (default is False).
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)

    def query_to_dataframe(self, query):
        """
        Executes a SQL query and returns the result as a pandas DataFrame.

        Parameters
        ----------
        query : str
            The SQL query to execute.

        Returns
        -------
        DataFrame
            The result of the query as a pandas DataFrame.
        """
        return pd.read_sql_query(text(query), self.engine)

    # def left_join_two_tables(
    #     self,
    #     table1,
    #     table2,
    #     table1_id,
    #     table2_id,
    #     select_columns=None,
    #     where_conditions=None,
    # ):
    #     """
    #     Performs a left join between two tables with optional select columns and where conditions.
    #
    #     Parameters
    #     ----------
    #     table1 : Base
    #         The first ORM table class.
    #     table2 : Base
    #         The second ORM table class.
    #     table1_id : str
    #         The column name in table1 to join on.
    #     table2_id : str
    #         The column name in table2 to join on.
    #     select_columns : list, optional
    #         List of columns to select (default is None).
    #     where_conditions : list, optional
    #         List of where conditions (default is None).
    #
    #     Returns
    #     -------
    #     list
    #         List of resulting rows after the join.
    #
    #     Examples
    #     --------
    #     >>> from models import NetflixMovie, AnotherTable
    #     >>> qm = QueryManager()
    #     >>> select_columns = [NetflixMovie.title, AnotherTable.name]
    #     >>> where_conditions = [NetflixMovie.show_id > 1]
    #     >>> joined_result = qm.left_join_two_tables(NetflixMovie, AnotherTable, 'show_id', 'related_id', select_columns, where_conditions)
    #     >>> for row in joined_result:
    #     ...     print(row)
    #     """
    #     query = self.session.query(table1).outerjoin(
    #         table2, getattr(table1, table1_id) == getattr(table2, table2_id)
    #     )
    #
    #     if select_columns:
    #         query = query.with_entities(*select_columns)
    #     if where_conditions:
    #         query = query.filter(and_(*where_conditions))
    #
    #     result = query.all()
    #     return result
    #
    # def left_join_three_tables(
    #     self,
    #     table1,
    #     table2,
    #     table3,
    #     table1_id,
    #     table2_id,
    #     table3_id,
    #     select_columns=None,
    #     where_conditions=None,
    # ):
    #     """
    #     Performs a left join between three tables with optional select columns and where conditions.
    #
    #     Parameters
    #     ----------
    #     table1 : Base
    #         The first ORM table class.
    #     table2 : Base
    #         The second ORM table class.
    #     table3 : Base
    #         The third ORM table class.
    #     table1_id : str
    #         The column name in table1 to join on.
    #     table2_id : str
    #         The column name in table2 to join on.
    #     table3_id : str
    #         The column name in table3 to join on.
    #     select_columns : list, optional
    #         List of columns to select (default is None).
    #     where_conditions : list, optional
    #         List of where conditions (default is None).
    #
    #     Returns
    #     -------
    #     list
    #         List of resulting rows after the join.
    #
    #     Examples
    #     --------
    #     >>> from models import NetflixMovie, AnotherTable, ThirdTable
    #     >>> qm = QueryManager()
    #     >>> select_columns = [NetflixMovie.title, AnotherTable.name, ThirdTable.description]
    #     >>> where_conditions = [NetflixMovie.show_id > 1, ThirdTable.related_id == 1]
    #     >>> joined_result = qm.left_join_three_tables(NetflixMovie, AnotherTable, ThirdTable, 'show_id', 'related_id', 'related_id', select_columns, where_conditions)
    #     >>> for row in joined_result:
    #     ...     print(row)
    #     """
    #     query = (
    #         self.session.query(table1)
    #         .outerjoin(table2, getattr(table1, table1_id) == getattr(table2, table2_id))
    #         .outerjoin(table3, getattr(table1, table1_id) == getattr(table3, table3_id))
    #     )
    #
    #     if select_columns:
    #         query = query.with_entities(*select_columns)
    #     if where_conditions:
    #         query = query.filter(and_(*where_conditions))
    #
    #     result = query.all()
    #     return result

    def left_join_two_tables(
        self,
        table1,
        table2,
        table1_id,
        table2_id,
        select_columns=None,
        where_conditions=None,
    ):
        """
        Performs a left join between two tables with optional select columns and where conditions.

        Parameters
        ----------
        table1 : Base
            The first ORM table class.
        table2 : Base
            The second ORM table class.
        table1_id : str
            The column name in table1 to join on.
        table2_id : str
            The column name in table2 to join on.
        select_columns : list, optional
            List of columns to select (default is None).
        where_conditions : list, optional
            List of where conditions (default is None).

        Returns
        -------
        list
            List of resulting rows after the join.
        """
        query = self.session.query(table1).outerjoin(
            table2, getattr(table1, table1_id) == getattr(table2, table2_id)
        )

        if select_columns:
            query = query.with_entities(*select_columns)
        if where_conditions:
            query = query.filter(and_(*where_conditions))

        result = query.all()
        return result

    def left_join_three_tables(
        self,
        table1,
        table2,
        table3,
        table1_id,
        table2_id,
        table3_id,
        select_columns=None,
        where_conditions=None,
    ):
        """
        Performs a left join between three tables with optional select columns and where conditions.

        Parameters
        ----------
        table1 : Base
            The first ORM table class.
        table2 : Base
            The second ORM table class.
        table3 : Base
            The third ORM table class.
        table1_id : str
            The column name in table1 to join on.
        table2_id : str
            The column name in table2 to join on.
        table3_id : str
            The column name in table3 to join on.
        select_columns : list, optional
            List of columns to select (default is None).
        where_conditions : list, optional
            List of where conditions (default is None).

        Returns
        -------
        list
            List of resulting rows after the join.
        """
        query = (
            self.session.query(table1)
            .outerjoin(table2, getattr(table1, table1_id) == getattr(table2, table2_id))
            .outerjoin(table3, getattr(table1, table1_id) == getattr(table3, table3_id))
        )

        if select_columns:
            query = query.with_entities(*select_columns)
        if where_conditions:
            query = query.filter(and_(*where_conditions))

        result = query.all()
        return result

    def list_tables(self):
        """
        Returns a list of all table names in the database.

        Returns
        -------
        list
            A list of table names.
        """
        # Connect to the database
        connection = self.engine.connect()

        # Query to retrieve table names
        query = "SELECT TableName FROM DBC.Tables WHERE TableKind = 'T' AND DatabaseName = DATABASE;"

        # Execute the query
        result = connection.execute(query)

        # Fetch all table names from the result
        table_names = [row[0] for row in result.fetchall()]

        # Close the database connection
        connection.close()

        return table_names


from sqlalchemy import create_engine, MetaData, Table, Column, Integer
from sqlalchemy.ext.declarative import declarative_base

# Initialize the metadata object
metadata = MetaData()
# Define a base class for the models
Base = declarative_base()


def generate_class_from_table_name(table_name, engine):
    """
    Generate an SQLAlchemy class based on a given table name.

    Parameters:
        table_name (str): The name of the table.
        engine (sqlalchemy.engine.base.Engine): The SQLAlchemy engine.

    Returns:
        type: The generated SQLAlchemy model class.
    """
    # Reflect the table from the database
    table = Table(table_name, metadata, autoload_with=engine)

    # Generate the class name based on the table name
    class_name = table_name.capitalize()

    # Check if the table has a primary key
    if not any(col.primary_key for col in table.columns):
        # Add an auto-incrementing ID column if no primary key exists
        table.append_column(Column("id", Integer, primary_key=True, autoincrement=True))

    # Create the class attributes
    class_attrs = {"__tablename__": table_name, "__table__": table}
    # Generate the class
    new_class = type(class_name, (Base,), class_attrs)

    # Log the generated class
    print(f"Class for table '{table_name}' is {new_class}.")
    return new_class


# Example usage:
# engine = create_engine("teradatasqlalchemy://user:password@host/")
# MyClass = generate_class_from_table_name("my_table", engine)
