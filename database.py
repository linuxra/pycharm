from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from configal import DATABASE_URI


class Database:
    def __init__(self):
        self.engine = create_engine(DATABASE_URI)
        self.Session = sessionmaker(bind=self.engine)

    def get_engine(self):
        return self.engine

    def get_session(self):
        return self.Session()
