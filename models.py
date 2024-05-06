from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class NetflixMovie(Base):
    __tablename__ = "netflix_movies"
    show_id = Column(Integer, primary_key=True)
    title = Column(String)
    director = Column(String)

    def __repr__(self):
        return f"<NetflixMovie(show_id={self.show_id}, title={self.title})>"


class AnotherTable(Base):
    __tablename__ = "another_table"
    id = Column(Integer, primary_key=True)
    related_id = Column(Integer)
    name = Column(String)

    def __repr__(self):
        return f"<AnotherTable(id={self.id}, related_id={self.related_id}, name={self.name})>"


class ThirdTable(Base):
    __tablename__ = "third_table"
    id = Column(Integer, primary_key=True)
    related_id = Column(Integer)
    description = Column(String)

    def __repr__(self):
        return f"<ThirdTable(id={self.id}, related_id={self.related_id}, description={self.description})>"
