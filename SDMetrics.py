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
