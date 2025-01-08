import os
import pandas as pd


def process_parquet_files(directory, column_name, value_to_remove):
    """
    Processes each Parquet file in the specified directory. Filters rows based on the provided column value condition,
    prints the filtered rows, and updates the Parquet file by removing these rows and resetting the index.

    Args:
    directory (str): The directory containing Parquet files.
    column_name (str): The column to apply the filter to.
    value_to_remove (str): The value to filter and remove from the column.

    Returns:
    None
    """
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            file_path = os.path.join(directory, filename)
            try:
                # Read the Parquet file
                df = pd.read_parquet(file_path)

                # Check if the column exists in the DataFrame
                if column_name in df.columns:
                    # Filter the DataFrame based on the condition
                    condition = df[column_name] == value_to_remove
                    filtered_df = df[condition]

                    # Check if there are any rows to remove
                    if not filtered_df.empty:
                        print(f"Filtered rows in {filename}:")
                        print(filtered_df)

                        # Remove the filtered rows from the DataFrame
                        df = df[~condition]

                        # Reset the index after removing rows
                        df.reset_index(drop=True, inplace=True)

                        # Save the updated DataFrame back to Parquet
                        df.to_parquet(file_path, engine='pyarrow')
                        print(f"Updated {filename} after removing filtered rows and resetting index.")
                    else:
                        print(
                            f"No rows to remove in {filename} based on the condition '{column_name} == {value_to_remove}'.")
                else:
                    print(f"Column '{column_name}' does not exist in {filename}.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage:
# Assume the directory is 'data' and you want to remove rows where a specific column has 'SomeVAR'
process_parquet_files('data', 'Column_Name', 'SomeVAR')

