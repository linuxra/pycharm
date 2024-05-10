from datetime import datetime


class SQLQueryGenerator:
    def __init__(self, start_date, end_date):
        """
        Initializes the SQLQueryGenerator with start and end dates.

        Args:
            start_date (str): Start date in the format 'YYYY-MM-DD'.
            end_date (str): End date in the format 'YYYY-MM-DD'.
        """
        # Validate date format at initialization
        self.validate_dates(start_date, end_date)
        self.start_date = start_date
        self.end_date = end_date

    def validate_dates(self, start_date, end_date):
        """Validates that the provided dates are in the correct format."""
        date_format = "%Y-%m-%d"
        try:
            datetime.strptime(start_date, date_format)
            datetime.strptime(end_date, date_format)
        except ValueError:
            raise ValueError(f"Dates must be in the format YYYY-MM-DD")

    def sql_for_sales(self):
        """Generates SQL for the sales segment."""
        return f"SELECT * FROM sales_table WHERE sale_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

    def sql_for_marketing(self):
        """Generates SQL for the marketing segment."""
        return f"SELECT * FROM marketing_table WHERE campaign_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

    def sql_for_finance(self):
        """Generates SQL for the finance segment."""
        return f"SELECT * FROM finance_table WHERE transaction_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

    def sql_for_hr(self):
        """Generates SQL for the HR segment."""
        return f"SELECT * FROM employee_table WHERE hire_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

    def sql_for_it(self):
        """Generates SQL for the IT segment."""
        return f"SELECT * FROM it_assets_table WHERE update_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

    def generate_query(self, model_segment):
        """
        Dispatches the correct SQL generation method based on the model segment.

        Args:
            model_segment (str): The business model segment.

        Returns:
            str: SQL query string.

        Raises:
            ValueError: If no method matches the provided model segment.
        """
        segment_method = {
            "sales": self.sql_for_sales,
            "marketing": self.sql_for_marketing,
            "finance": self.sql_for_finance,
            "hr": self.sql_for_hr,
            "it": self.sql_for_it,
        }.get(model_segment)

        if segment_method is not None:
            return segment_method()
        else:
            raise ValueError(f"No SQL method found for model segment: {model_segment}")


# Example usage:
try:
    query_generator = SQLQueryGenerator("2023-05-01", "2023-05-31")
    sql_query = query_generator.generate_query("finance")
    print(sql_query)
except ValueError as e:
    print(e)
import pandas as pd
from sqlalchemy import create_engine
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class Config:
    model_segment: str
    yyyymm: str
    base_path: Path = field(default_factory=lambda: Path("/path/to/data/"))
    extension: str = (
        ".parquet"  # Changed to .parquet for saving DataFrame in Parquet format
    )
    file_path: Path = field(init=False)
    business_area: str = field(init=False)
    PSI_FILE_NAME: Path = field(init=False)  # PSI file name

    def __post_init__(self):
        self.validate_date_format()
        self.file_path = (
            self.base_path / f"{self.model_segment}_{self.yyyymm}{self.extension}"
        )
        self.business_area = self.determine_business_area()
        self.PSI_FILE_NAME = self.generate_file_name("PSI")

    def validate_date_format(self):
        try:
            datetime.strptime(self.yyyymm, "%Y%m")
        except ValueError:
            raise ValueError("yyyymm must be in YYYYMM format")

    def determine_business_area(self):
        mapping = {
            "sales": "Sales and Marketing",
            "marketing": "Marketing Department",
            "finance": "Financial Operations",
            "hr": "Human Resources",
            "it": "Information Technology",
        }
        return mapping.get(self.model_segment, "Unknown Business Area")

    def generate_file_name(self, file_type):
        return (
            self.base_path
            / f"{self.model_segment}_{self.yyyymm}_{file_type}{self.extension}"
        )


class SQLQueryGenerator:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def generate_query(self, model_segment):
        segment_query = {
            "sales": f"SELECT * FROM sales_table WHERE sale_date BETWEEN '{self.start_date}' AND '{self.end_date}'",
            "marketing": f"SELECT * FROM marketing_table WHERE campaign_date BETWEEN '{self.start_date}' AND '{self.end_date}'",
            "finance": f"SELECT * FROM finance_table WHERE transaction_date BETWEEN '{self.start_date}' AND '{self.end_date}'",
            "hr": f"SELECT * FROM employee_table WHERE hire_date BETWEEN '{self.start_date}' AND '{self.end_date}'",
            "it": f"SELECT * FROM it_assets_table WHERE update_date BETWEEN '{self.start_date}' AND '{self.end_date}'",
        }.get(model_segment)

        if segment_query is None:
            raise ValueError(
                f"No SQL template found for model segment: {model_segment}"
            )
        return segment_query


# Initialize Config and Query Generator
config = Config(model_segment="finance", yyyymm="202305")
query_gen = SQLQueryGenerator("2023-05-01", "2023-05-31")
sql_query = query_gen.generate_query(config.model_segment)

# Database connection string (example)
connection_url = (
    "teradata+teradatasqlalchemy://<username>:<password>@<hostname>/<database>"
)

# Create engine
engine = create_engine(connection_url)
logging.info("Database engine created.")

# Test connection and fetch data
try:
    with engine.connect() as connection:
        logging.info("Database connection successfully established.")
        df = pd.read_sql(sql_query, connection)
        logging.info(f"Data fetched successfully with {len(df)} rows.")

        # Save the DataFrame using Config generated file path
        df.to_parquet(config.file_path, index=False)
        logging.info(f"Data saved to {config.file_path}")
except Exception as e:
    logging.error(
        "Failed to connect to the database or execute the query", exc_info=True
    )
    raise e
