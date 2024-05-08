from dataclasses import dataclass, field, asdict
import pandas as pd


@dataclass
class DataMetrics:
    df: pd.DataFrame
    target: str = "bad"
    phat: str = "fico"
    risk_ordinality: str = "decreasing"
    target_fico: str = "medium"
    target_score: str = "pass"
    another_column: str = "some_value"
    sd: float = field(init=False)

    def __post_init__(self):
        # Calculate SD when the class instance is created
        self.sd = self.calc_sd()

    def calc_sd(self):
        # Placeholder for the actual SD calculation function
        # You would replace this with your actual calculation logic
        return calc_sd(self.df, self.phat, self.target)


def create_sd_csv(metrics: DataMetrics, filename: str):
    """
    Create SD CSV file using data from a DataMetrics instance.

    Parameters:
        metrics (DataMetrics): Instance of DataMetrics containing all necessary data.
        filename (str): Output file name.
    """
    # Convert the dataclass instance to a dictionary
    data_dict = asdict(metrics)

    # Remove the 'df' key if it's included, as it's not meant to be saved to CSV
    data_dict.pop("df", None)

    # Create DataFrame from the dictionary and save to CSV
    result_df = pd.DataFrame([data_dict])
    result_df.to_csv(filename, index=False)


# Example usage
# Assuming df is your DataFrame and calc_sd is a function defined elsewhere
# metrics = DataMetrics(df)
# create_sd_csv(metrics, 'output_sd.csv')


from dataclasses import dataclass, field, asdict
import pandas as pd


@dataclass
class DataMetrics:
    """
    A data class to manage and compute metrics for a given dataset.

    Attributes:
        df (pd.DataFrame): The data frame containing the dataset.
        target (str): The name of the target column used for certain calculations.
        phat (str): The name of the probability estimate column.
        risk_ordinality (str): Describes the risk ordinality ('increasing' or 'decreasing').
        target_fico (str): A descriptor for the FICO score range.
        target_score (str): Pass or fail status based on certain criteria.
        another_column (str): An additional column for extra data or metadata.
        sd (float): Standard deviation, computed post-initialization. Not initialized directly.
    """

    df: pd.DataFrame
    target: str
    phat: str
    risk_ordinality: str
    target_fico: str
    target_score: str
    another_column: str
    sd: float = field(init=False)

    def __post_init__(self):
        """Post-initialization to compute the standard deviation."""
        self.validate_data()
        self.sd = self.calc_sd()

    def validate_data(self):
        """
        Validates the presence of necessary columns in the DataFrame.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        required_columns = {self.phat, self.target}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def calc_sd(self):
        """
        Calculates the standard deviation for the dataset using specified columns.

        Returns:
            float: The computed standard deviation.

        Note:
            Replace this placeholder with the actual calculation logic.
        """
        # Placeholder function for actual SD calculation
        return calc_sd(self.df, self.phat, self.target)

    @classmethod
    def create_variant_one(cls, df):
        """
        Factory method to create an instance of DataMetrics with a predefined configuration.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            DataMetrics: An instance with a specific configuration.
        """
        return cls(df, "bad", "fico", "decreasing", "medium", "pass", "some_value")

    @classmethod
    def create_variant_two(cls, df):
        """
        Factory method to create an instance of DataMetrics with a second predefined configuration.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            DataMetrics: An instance with an alternative specific configuration.
        """
        return cls(df, "medium", "score", "increasing", "high", "fail", "another_value")


def create_sd_csv(metrics: DataMetrics, filename: str):
    """
    Generates a CSV file from the data within a DataMetrics instance.

    Parameters:
        metrics (DataMetrics): The instance containing the data to be saved.
        filename (str): The name of the file to save the data to.
    """
    data_dict = asdict(metrics)
    data_dict.pop(
        "df", None
    )  # Remove the DataFrame from the dictionary as it's not serializable
    result_df = pd.DataFrame([data_dict])
    result_df.to_csv(filename, index=False)


# Example usage:
df = pd.DataFrame()  # Assume df is properly set up
try:
    metrics_one = DataMetrics.create_variant_one(df)
    create_sd_csv(metrics_one, "output_one_sd.csv")
except ValueError as e:
    print(e)
