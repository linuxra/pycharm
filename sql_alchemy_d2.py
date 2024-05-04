from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

Base = declarative_base()


class NetflixMovie(Base):
    __tablename__ = "netflix_movies"
    show_id = Column(Integer, primary_key=True)
    title = Column(String)
    director = Column(String)
    release_year = Column(Date)
    duration = Column(String)


DATABASE_URI = "postgresql://postgres:temp123@localhost:5433/postgres"
engine = create_engine(DATABASE_URI)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)


def fetch_netflix_movies():
    session = DBSession()
    netflix_movies_df = (
        pd.DataFrame()
    )  # Initialize DataFrame to ensure it's always defined
    try:
        query = session.query(NetflixMovie)
        netflix_movies_df = pd.read_sql(query.statement, session.bind)
        print(netflix_movies_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        session.close()
    return netflix_movies_df


if __name__ == "__main__":
    netflix_movies_df = fetch_netflix_movies()
