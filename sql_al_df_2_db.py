import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the base class from which all mapped classes should inherit
Base = declarative_base()


def load_dataframe_to_sql(df, table_name, database_uri, if_exists="replace"):
    """
    Loads a pandas DataFrame into a PostgreSQL database table.
    """
    engine = create_engine(database_uri)
    if if_exists == "replace":
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    df.to_sql(table_name, con=engine, index=False, if_exists=if_exists)
    print(f"Data has been loaded into '{table_name}' successfully.")


# Example Usage
DATABASE_URI = "postgresql://postgres:temp123@localhost:5433/postgres"

# Load data from CSV
df = pd.read_csv(
    "/Users/rkaddanki/Downloads/bank.csv"
)  # Update the path to your CSV file

# Load the DataFrame into the SQL table
load_dataframe_to_sql(df, "bank", DATABASE_URI, if_exists="replace")
