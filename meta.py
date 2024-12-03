import pyarrow.parquet as pq

# Path to your Parquet file
file_path = 'path_to_your_parquet_file.parquet'

# Read the Parquet file metadata
parquet_file = pq.ParquetFile(file_path)

# Get the schema from the Parquet file
schema = parquet_file.schema

# Print each field in the schema
for field in schema:
    print(f"Column Name: {field.name}")
    print(f"Data Type: {field.type}")
    print(f"Nullable: {field.nullable}")
    print(f"Metadata: {field.metadata}")
    print("---")