import pandas as pd
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
df = pd.read_csv("./../data/netflix_titles.csv")

# Display the DataFrame
print(df)

# Connect to the PostgreSQL server
engine = create_engine('postgresql://postgres:temp123@localhost/postgres')

# Save the DataFrame to the PostgreSQL table
df.to_sql('netflix_movies', engine, if_exists='replace', index=False)
query = """select * from netflix_movies limit 10"""
result_df = pd.read_sql(query, engine)
print(" Table rows")
print(result_df.head(10))

# Close the connection
engine.dispose()
