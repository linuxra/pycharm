import polars as pl

df = pl.read_csv("/Users/rkaddanki/Downloads/train.csv")
print(df.head())
