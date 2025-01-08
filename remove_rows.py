import os
import pandas as pd


def process_parquet_files(directory, condition):
    """
    Processes each Parquet file in the specified directory. Filters rows based on the provided condition,
    prints the filtered rows, and updates the Parquet file by removing these rows.

    Args:
    directory (str): The directory containing Parquet files.
    condition (str): A string representing the filtering condition (e.g., 'Age > 30').

    Returns:
    None
    """
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            file_path = os.path.join(directory, filename)
            # Read the Parquet file
            df = pd.read_parquet(file_path)

            # Filter the DataFrame based on the condition
            filtered_df = df.query(condition)

            # Check if there are any rows to remove
            if not filtered_df.empty:
                print(f"Filtered rows in {filename}:")
                print(filtered_df)

                # Remove the filtered rows from the DataFrame
                df = df.drop(filtered_df.index)

                # Save the updated DataFrame back to Parquet
                df.to_parquet(file_path, engine="pyarrow")
                print(f"Updated {filename} after removing filtered rows.")
            else:
                print(f"No rows to remove in {filename} based on the condition.")


# Example usage:
# Assume the directory is 'data' and you want to remove rows where 'Age' is greater than 30
process_parquet_files("data", "Age > 30")
