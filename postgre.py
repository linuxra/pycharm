import psycopg2

connection = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="temp123"
)

cursor = connection.cursor()

query = '''
WITH rank_numbers AS (
  SELECT * FROM (VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10)) AS t(rank)
),
counter_numbers AS (
  SELECT * FROM (VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21), (22), (23), (24)) AS t(counter)
),
all_combinations AS (
  SELECT rank, counter
  FROM rank_numbers
  CROSS JOIN counter_numbers
),
dummy_data AS (
  SELECT * FROM (VALUES 
    (1, 1, 5),
    (1, 3, 3),
    (2, 2, 8),
    (3, 1, 4),
    (4, 5, 1),
    (10, 24, 2)
  ) AS t(rank, counter, badcount)
)

SELECT ac.rank, ac.counter, COALESCE(dd.badcount, 0) AS badcount
FROM all_combinations AS ac
LEFT JOIN dummy_data AS dd
  ON ac.rank = dd.rank
  AND ac.counter = dd.counter
ORDER BY ac.rank, ac.counter;
'''

cursor.execute(query)
results = cursor.fetchall()

for row in results:
    print(row)

cursor.close()
connection.close()
