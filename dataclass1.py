from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Config:
    model_segment: str
    yyyymm: str
    base_path: Path = field(default_factory=lambda: Path("/path/to/data/"))
    extension: str = ".csv"
    file_path: Path = field(init=False)
    business_area: str = field(init=False)
    PSI_FILE_NAME: Path = field(init=False)  # PSI file name

    def __post_init__(self):
        # Validate the yyyymm format
        self.validate_date_format()

        # Set the full file path using pathlib
        self.file_path = (
            self.base_path / f"{self.model_segment}_{self.yyyymm}{self.extension}"
        )

        # Determine business area based on model segment
        self.business_area = self.determine_business_area()

        # Set the PSI file name
        self.PSI_FILE_NAME = self.generate_file_name("PSI")

    def validate_date_format(self):
        """Ensure yyyymm is a valid date string in the format YYYYMM."""
        try:
            datetime.strptime(self.yyyymm, "%Y%m")
        except ValueError:
            raise ValueError("yyyymm must be in YYYYMM format")

    def determine_business_area(self):
        """Map model segment to business area."""
        mapping = {
            "sales": "Sales and Marketing",
            "marketing": "Marketing Department",
            "finance": "Financial Operations",
            "hr": "Human Resources",
            "it": "Information Technology",
        }
        return mapping.get(self.model_segment, "Unknown Business Area")

    def generate_file_name(self, file_type):
        """Generate a file name based on the type, model segment, and date."""
        return (
            self.base_path
            / f"{self.model_segment}_{self.yyyymm}_{file_type}{self.extension}"
        )


# Usage example:
try:
    config = Config(model_segment="sales", yyyymm="202305")
    print(f"The data file path is: {config.file_path}")
    print(f"Business Area: {config.business_area}")
    print(f"PSI File Name: {config.PSI_FILE_NAME}")
except ValueError as e:
    print(e)
database:
  host: your_db_host
  user: your_db_user
  password: your_db_password
  database: your_database

logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

cache:
  expiry_seconds: 3600

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import yaml
import os
from functools import lru_cache
from datetime import datetime, timedelta
from utils import OmDataPull, DateUtils, TeraConnector, Timer

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configure logging
logging.basicConfig(level=config['logging']['level'], format=config['logging']['format'])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def get_initial_data(dt: Any, vintage: int, logger: logging.Logger) -> Tuple[OmDataPull, Any, Any, str, str, str]:
    """
    Retrieves initial data and formats required for processing.

    Parameters:
        dt (datetime): Reference date for pulling data.
        vintage (int): Vintage period in months.
        logger (Logger): Logger object for logging information.

    Returns:
        tuple: Contains datapull, start_date, end_date, score_yyyymm, perf_yyyymm, vymm
    """
    try:
        logger.info("Retrieving initial data and date formats.")
        datapull = OmDataPull(logger)
        start_date, end_date, score_yyyymm, perf_yyyymm = DateUtils.get_date_info(dt, vintage)
        vymm = DateUtils.convert_date_format(start_date)
        return datapull, start_date, end_date, score_yyyymm, perf_yyyymm, vymm
    except Exception as e:
        logger.error(f"Error in get_initial_data: {e}")
        raise


def execute_query(connector: TeraConnector, sql: str) -> pd.DataFrame:
    """
    Executes an SQL query and returns the result as a DataFrame.

    Parameters:
        connector (TeraConnector): Connector object for database operations.
        sql (str): SQL query string.

    Returns:
        DataFrame: Result of the SQL query.
    """
    try:
        logger.info(f"Executing query: {sql}")
        return connector.execute_query(sql, as_dataframe=True)
    except Exception as e:
        connector.logger.error(f"Error in execute_query: {e}")
        raise


def process_short_vintage(datapull: OmDataPull, start_date: Any, vintage: int, score_yyyymm: str, vymm: str,
                          df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Processes data for short vintage periods (less than 24 months).

    Parameters:
        datapull (OmDataPull): Data pull object for SQL generation.
        start_date (datetime): Start date for the data pull.
        vintage (int): Vintage period in months.
        score_yyyymm (str): Score year and month in YYYYMM format.
        vymm (str): Formatted start date.
        df (DataFrame): Initial DataFrame to be processed.
        logger (Logger): Logger object for logging information.

    Returns:
        DataFrame: Processed DataFrame with performance metrics.
    """
    try:
        logger.info("Processing short vintage data.")
        perf_mons = DateUtils.generate_next_months(start_date, vintage)[-1]
        case_statements_dict = datapull.generate_monthly_case_statement()
        sql_query = datapull.acq_gen_query(case_statements_dict, vintage, score_yyyymm, perf_mons)
        with Timer(logger):
            df["score_vymm"] = vymm
            df["perf_yyyymm"] = perf_mons
        return df
    except Exception as e:
        logger.error(f"Error in process_short_vintage: {e}")
        raise


def query_parallel(connector: TeraConnector, sql: str) -> pd.DataFrame:
    """
    Executes an SQL query in parallel and returns the result as a DataFrame.

    Parameters:
        connector (TeraConnector): Connector object for database operations.
        sql (str): SQL query string.

    Returns:
        DataFrame: Result of the SQL query.
    """
    try:
        logger.info(f"Executing parallel query: {sql}")
        return connector.query_teradata(sql)
    except Exception as e:
        connector.logger.error(f"Error in query_parallel: {e}")
        raise


def process_long_vintage(datapull: OmDataPull, start_date: Any, vintage: int, score_yyyymm: str, vymm: str,
                         connector: TeraConnector, df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes data for long vintage periods (24 months or more).

    Parameters:
        datapull (OmDataPull): Data pull object for SQL generation.
        start_date (datetime): Start date for the data pull.
        vintage (int): Vintage period in months.
        score_yyyymm (str): Score year and month in YYYYMM format.
        vymm (str): Formatted start date.
        connector (TeraConnector): Connector object for database operations.
        df (DataFrame): Initial DataFrame to be processed.

    Returns:
        DataFrame: Processed and merged DataFrame with performance metrics.
    """
    try:
        logger.info("Processing long vintage data.")
        perf_mons = DateUtils.generate_next_months(start_date, vintage)[-1]
        sql1 = datapull.acq_sql_gen(score_yyyymm, perf_mons)

        perf_mons2 = DateUtils.generate_next_months(start_date, 12)[-1]
        sql2 = datapull.acq_sql_gen2(vymm, perf_mons2)

        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(query_parallel, connector, sql1)
            future2 = executor.submit(query_parallel, connector, sql2)
            df1 = future1.result()
            df2 = future2.result()

        joined_df = pd.merge(df1, df2, on="APP_REF_NO", how="inner")
        joined_df["bad"] = np.where(
            (joined_df["DLO_90DP_Ever_1"] == 1) | (joined_df["DIQ_90DP_Ever_2"] == 1),
            1,
            0
        )
        return joined_df
    except Exception as e:
        connector.logger.error(f"Error in process_long_vintage: {e}")
        raise


def acq_pull_perf(bus_area: str, dt: Any, vintage: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Pulls acquisition performance data for a specified business area and date range.

    Parameters:
        bus_area (str): Business area identifier.
        dt (datetime): Reference date for pulling data.
        vintage (int): Vintage period in months.
        logger (Logger): Logger object for logging information.

    Returns:
        DataFrame: The processed DataFrame with performance metrics.
    """
    try:
        logger.info(f"Starting data pull for business area: {bus_area}, date: {dt}, vintage: {vintage}")
        datapull, start_date, end_date, score_yyyymm, perf_yyyymm, vymm = get_initial_data(dt, vintage, logger)

        with TeraConnector(logger) as connector:
            sql = datapull.acq_perf_accts(start_date, end_date, score_yyyymm)
            df = execute_query(connector, sql)

            if vintage < 24:
                df = process_short_vintage(datapull, start_date, vintage, score_yyyymm, vymm, df, logger)
            else:
                df = process_long_vintage(datapull, start_date, vintage, score_yyyymm, vymm, connector, df)

        logger.info("Data pull completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in acq_pull_perf: {e}")
        raise
