import os


def rename_files(directory):
    """
    Renames files in the specified directory from 'M502225_YYYYMMDD.parquet' to 'M502225_lqr_YYYYMMDD.parquet'.

    Args:
    directory (str): The directory where the files are located.
    """
    # Navigate to the directory
    os.chdir(directory)

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the expected pattern
        if filename.startswith("M502225_") and filename.endswith(".parquet"):
            date_part = filename[8:-8]  # Extract YYYYMMDD from the filename
            # Check if the extracted part is a valid date
            if date_part.isdigit() and len(date_part) == 8:
                new_filename = f"M502225_lqr_{date_part}.parquet"
                # Rename the file
                os.rename(filename, new_filename)
                print(f"Renamed {filename} to {new_filename}")
            else:
                print(f"Skipping {filename} - does not match date format")
        else:
            print(f"Skipping {filename} - does not match the expected pattern")


# Usage example:
# Specify the directory containing the files
directory_path = "/path/to/your/files"
rename_files(directory_path)
