from sql_alchemy_class import QueryManager
from models import NetflixMovie, AnotherTable, ThirdTable
import pandas as pd


def main():
    qm = QueryManager()

    # Example 1: List tables
    print("Tables in the database:")
    print(qm.list_tables())

    # Example 2: Get Netflix movies
    print("First 3 Netflix movies:")
    for movie in qm.get_netflix_movies(limit=3):
        print(f"{movie.title}, {movie.director}")

    # Example 3: Introspect table
    print("Columns in 'netflix_movies' table:")
    for column in qm.introspect_table("netflix_movies"):
        print(f"{column['name']}, {column['type']}")

    # Example 4: Execute raw query
    print("Execute raw query:")
    raw_result = qm.execute_raw_query("SELECT * FROM netflix_movies LIMIT 3")
    for row in raw_result:
        print(row)

    # Example 5: Upload DataFrame
    df = pd.DataFrame(
        {
            "show_id": [100, 101],
            "title": ["Movie 100", "Movie 101"],
            "director": ["Director 100", "Director 101"],
        }
    )
    qm.upload_dataframe(df, "netflix_movies_test")
    print("Uploaded DataFrame to 'netflix_movies_test' table.")
    qm.upload_dataframe(df, "another_table")
    print("Uploaded DataFrame to 'netflix_movies_test' table.")
    # Example 6: Query to DataFrame
    print("Query to DataFrame:")
    df_result = qm.query_to_dataframe("SELECT * FROM netflix_movies LIMIT 3")
    print(df_result)
    bank = pd.read_csv("/Users/rkaddanki/Downloads/bank.csv")
    qm.upload_dataframe(bank, "bank")
    print("Uploaded DataFrame to 'bank' table.")
    print("Bank DataFrame:")
    print(bank.head(5))


if __name__ == "__main__":
    main()
