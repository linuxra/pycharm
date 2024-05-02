# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.orm import sessionmaker, declarative_base  # Updated import here
#
# # Replace these with your actual database credentials
# DATABASE_URI = "postgresql://postgres:temp123@localhost/postgres"
#
# Base = declarative_base()  # This usage is now correctly imported
#
#
# class Stacker(Base):
#     __tablename__ = "stackover"
#     PostId = Column(Integer, primary_key=True)
#     Title = Column(String)
#
#
# # Create engine
# engine = create_engine(DATABASE_URI)
#
# # Bind a session
# Session = sessionmaker(bind=engine)
# session = Session()
#
# # Query the database
# results = (
#     session.query(Stacker.PostId, Stacker.Title)
#     .filter(Stacker.Title.like("%calculate%"))
#     .limit(100)
#     .all()
# )
#
# # Print the results
# for result in results:
#     print(f"PostId: {result.PostId}, Title: {result.Title}")
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuration and model definition
DATABASE_URI = "postgresql://postgres:temp123@localhost/postgres"
Base = declarative_base()


class Stacker(Base):
    """Data model for the stackover table."""

    __tablename__ = "stackover"
    PostId = Column(Integer, primary_key=True)
    Title = Column(String)

    def __repr__(self):
        return f"<Stacker(PostId={self.PostId}, Title={self.Title})>"


# Database setup
engine = create_engine(DATABASE_URI)
Base.metadata.create_all(engine)  # Create tables based on the Base metadata
Session = sessionmaker(bind=engine)
session = Session()


def main():
    """Main function to perform query and display results."""
    try:
        # Query to select PostId and Title where Title contains 'calculate'
        results = (
            session.query(Stacker.PostId, Stacker.Title)
            .filter(Stacker.Title.like("%calculate%"))
            .all()
        )

        # Print the results
        for result in results:
            print(f"PostId: {result.PostId}, Title: {result.Title}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure that the session is closed when done
        session.close()


if __name__ == "__main__":
    main()
