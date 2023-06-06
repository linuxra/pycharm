import psycopg2
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:temp123@localhost/postgres')
connection = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="temp123"
)
df = pd.DataFrame({
    'id1': [1, 1, 1, 2, 2, 3, 3],
    'id2': [5, None, 7, None, 4, None, None]
})
print(df)
cursor = connection.cursor()
df.to_sql('example_table', engine, if_exists='replace')
# This below sql selects max value for id2
# query = '''
# WITH CTE AS (
#   SELECT id1,
#          id2,
#          ROW_NUMBER() OVER(PARTITION BY id1 ORDER BY CASE WHEN id2 is NULL THEN 1 else 0 end, id2 DESC) as rn
#   FROM example_table
# )
# SELECT id1, id2
# FROM CTE
# WHERE rn = 1
#
#
# '''
query = '''
WITH CTE AS (
  SELECT id1, 
         id2, 
         ROW_NUMBER() OVER(PARTITION BY id1 ORDER BY id2 ASC) as rn
  FROM example_table
)
SELECT id1, id2
FROM CTE
WHERE rn = 1


'''
cursor.execute(query)
results = cursor.fetchall()

for row in results:
    print(row)

cursor.close()
connection.close()
