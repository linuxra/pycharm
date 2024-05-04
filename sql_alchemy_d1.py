from sqlalchemy import create_engine, text
from sqlalchemy import inspect

# Database URI
# DATABASE_URI = "postgresql://postgres:temp123@localhost/postgres"
DATABASE_URI = "postgresql://postgres:temp123@localhost:5432/postgres"

# Create engine
engine = create_engine(DATABASE_URI)

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class NetflixMovie(Base):
    __tablename__ = "netflix_movies"
    show_id = Column(Integer, primary_key=True)
    title = Column(String)
    director = Column(String)
    cast = Column(String)
    country = Column(String)
    date_added = Column(String)
    release_year = Column(Integer)
    rating = Column(String)
    duration = Column(String)
    listed_in = Column(String)
    description = Column(String)


# Create the engine


# Create a configured "Session" class
Session = sessionmaker(bind=engine)

# Create a session
session = Session()

# Query the first 5 entries from the netflix_movies table
result = session.query(NetflixMovie).limit(5).all()

# Display the result
for row in result:
    print(row.title, row.director, row.release_year)


def list_tables():
    # Create an inspector object using sqlalchemy.inspect()
    inspector = inspect(engine)

    # Get list of tables
    tables = inspector.get_table_names()
    print("Tables in the database:")
    for table in tables:
        print(table)


from sqlalchemy import inspect

# Reflect the table
inspector = inspect(engine)
table_info = inspector.get_columns("netflix_movies")

# Display the table information
for column in table_info:
    print(f"Column: {column['name']}, Type: {column['type']}")


# Access the table metadata
table = NetflixMovie.__table__

# Display the column information
for column in table.columns:
    print(f"Column: {column.name}, Type: {column.type}")

# Introspect the properties of the NetflixMovie class
for prop in NetflixMovie.__mapper__.iterate_properties:
    print(f"Property: {prop.key}, Type: {type(prop)}")


from sqlalchemy import create_engine, MetaData, Table, Column, Integer, text
from sqlalchemy.orm import declarative_base


metadata = MetaData()

Base = declarative_base()


# Function to generate the class
def generate_class_from_table_name(table_name):
    # Reflect the table from the database
    table = Table(table_name, metadata, autoload_with=engine)

    # Generate the class
    class_name = table_name.capitalize()
    if not any(col.primary_key for col in table.columns):
        table.append_column(Column("id", Integer, primary_key=True, autoincrement=True))

    class_attrs = {"__tablename__": table_name, "__table__": table}
    new_class = type(class_name, (Base,), class_attrs)

    print(f"Class for table '{table_name}' is {new_class}.")
    return new_class


# Generate the class for a specific table
NetflixMovie = generate_class_from_table_name("netflix_movies")
# print(NetflixMovie.__str__() + " is the class for the table netflix_movies.")

# Example usage
with engine.connect() as conn:
    result = conn.execute(text(f"SELECT * FROM {NetflixMovie.__tablename__} LIMIT 5"))
    for row in result:
        print(row)


if __name__ == "__main__":
    list_tables()
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM netflix_movies LIMIT 5"))
        print("ok")
        for row in result:
            print(row)
