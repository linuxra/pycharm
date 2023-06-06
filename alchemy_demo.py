from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# replace these with your actual database credentials
DATABASE_URI = 'postgresql://postgres:temp123@localhost/postgres'

Base = declarative_base()

class Stacker(Base):
    __tablename__ = 'stackover'

    PostId = Column(Integer, primary_key=True)
    Title = Column(String)

# create engine
engine = create_engine(DATABASE_URI)

# bind a session
Session = sessionmaker(bind=engine)
session = Session()

# query the database
results = session.query(Stacker.PostId, Stacker.Title)\
    .filter(Stacker.Title.like('%calculate%'))\
    .limit(100).all()

# print the results
for result in results:
    print(f'PostId: {result.PostId}, Title: {result.Title}')
