df2 = df2.set_index('r')

# Mapping of cb_cd values to column names
cb_cd_mapping = {'ex': 'ex_cd', 'tu': 'tu_cd', 'ef': 'ef_cd'}

# Replace values in 'r1', 'r2', 'r3', 'r4' based on the merged values and cb_cd
for col in ['r1', 'r2', 'r3', 'r4']:
    # Merge with df2
    df1 = df1.merge(df2, left_on=col, right_index=True, suffixes=('', f'_{col}'))

    # Replace values
    df1[col] = df1.apply(lambda row: row[cb_cd_mapping[row['cb_cd']] + f'_{col}'], axis=1)

    # Drop the merged columns
    df1.drop([cb_cd_mapping[key] + f'_{col}' for key in cb_cd_mapping.keys()], axis=1, inplace=True)

# Display the modified DataFrame
print(df1)


WITH unpivoted AS (
    SELECT
        id,
        cb_cd,
        CASE col
            WHEN 'r1' THEN r1
            WHEN 'r2' THEN r2
            WHEN 'r3' THEN r3
            WHEN 'r4' THEN r4
        END AS r,
        col
    FROM
        table1
    CROSS JOIN (
        SELECT 'r1' AS col
        UNION ALL SELECT 'r2'
        UNION ALL SELECT 'r3'
        UNION ALL SELECT 'r4'
    ) AS columns
),
joined AS (
    SELECT
        u.id,
        u.col,
        CASE
            WHEN u.cb_cd = 'TU' THEN t2.TU_rsn_cd
            WHEN u.cb_cd = 'EX' THEN t2.EXP_rsn_cd
            WHEN u.cb_cd = 'EF' THEN t2.EFX_rsn_cd
            ELSE u.r
        END AS r_new
    FROM
        unpivoted u
    LEFT JOIN
        table2 t2 ON u.r = t2.r
)
SELECT
    id,
    MAX(CASE WHEN col = 'r1' THEN r_new END) AS r1,
    MAX(CASE WHEN col = 'r2' THEN r_new END) AS r2,
    MAX(CASE WHEN col = 'r3' THEN r_new END) AS r3,
    MAX(CASE WHEN col = 'r4' THEN r_new END) AS r4
FROM
    joined
GROUP BY
    id;
