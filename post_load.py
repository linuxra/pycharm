import pandas as pd
from sqlalchemy import create_engine

import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Generate the data
ranks = list(range(1, 11))
counters = list(range(1, 25))
data = []

for rank in ranks:
    for counter in counters:
        bad = np.random.randint(1, 100)  # Generates a random integer between 1 and 100
        data.append([rank, counter, bad])

# Create the DataFrame
df = pd.DataFrame(data, columns=["rank", "counter", "bad"])

# Display the DataFrame
print(df)

# Connect to the PostgreSQL server
engine = create_engine('postgresql://postgres:temp123@localhost/postgres')
query = """
SELECT
  t1.rank,
  t1.counter,
  t1.bad,
  SUM(t1.bad) OVER (PARTITION BY t1.rank ORDER BY t1.counter ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_bad
FROM rai AS t1
ORDER BY t1.rank, t1.counter;
"""
# Save the DataFrame to the PostgreSQL table
# df.to_sql('rai', engine, if_exists='replace', index=False)

result_df = pd.read_sql(query, engine)
print(" Table rows")
print(result_df.head(10))

# Close the connection
engine.dispose()
