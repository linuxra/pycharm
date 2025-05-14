import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

def read_sas_cols(path: str, cols: List[str]) -> pd.DataFrame:
    """
    Read only a subset of columns from a .sas7bdat file.
    (pandas.read_sas doesn’t support usecols, so we slice after reading.)
    """
    df = pd.read_sas(path, format='sas7bdat', encoding='utf-8')
    return df[cols].copy()

def filter_actual(df: pd.DataFrame, grp_col: str = "grp") -> pd.DataFrame:
    """Keep only rows where grp_col == 'actual'."""
    return df[df[grp_col] == "actual"].copy()

def parse_source_date(df: pd.DataFrame, src_col: str = "source") -> pd.DataFrame:
    """
    Convert a column of month_abbr_two-digit-year strings (e.g. 'apr_98')
    into a datetime column named 'source_date'.
    """
    df = df.copy()
    # %b = locale’s abbreviated month name, %y = two-digit year
    df['source_date'] = pd.to_datetime(df[src_col], format='%b_%y')
    return df

def compare_sas_newmean(
    dir_old: str,
    dir_new: str,
    key_cols: List[str],
    cols_to_read: List[str] = ["source", "grp", "newmean"],
    grp_col: str = "grp",
    src_col: str = "source"
):
    """
    For each .sas7bdat in dir_old, read only source/grp/newmean,
    filter grp='actual', parse source→date, merge on key_cols,
    compute diff in newmean, then sort & plot by that date.
    """
    for fname in os.listdir(dir_old):
        if not fname.lower().endswith(".sas7bdat"):
            continue

        old_path = os.path.join(dir_old, fname)
        new_path = os.path.join(dir_new, fname)
        if not os.path.exists(new_path):
            print(f"– skipping {fname!r}: no match in new directory.")
            continue

        # read only needed cols, filter to actual
        df_old = filter_actual(read_sas_cols(old_path, cols_to_read), grp_col)
        df_new = filter_actual(read_sas_cols(new_path, cols_to_read), grp_col)

        # parse the 'source' strings into a real datetime
        df_old = parse_source_date(df_old, src_col)
        df_new = parse_source_date(df_new, src_col)

        # merge on your key(s)
        merged = pd.merge(
            df_old, df_new,
            on=key_cols,
            suffixes=("_old", "_new"),
            how="inner"
        )

        # compute absolute and percent diff of newmean
        merged["diff_newmean"] = merged["newmean_new"] - merged["newmean_old"]
        merged["diff_newmean_pct"] = (
            merged["diff_newmean"]
            / merged["newmean_old"].replace({0: pd.NA})
            * 100
        )

        # sort by the parsed date
        merged.sort_values("source_date", inplace=True)

        print(f"\n— Results for {fname}:")
        print(merged[[*key_cols,
                      "newmean_old", "newmean_new",
                      "diff_newmean", "diff_newmean_pct",
                      "source_date"]])

        # plot percent difference in chronological order
        x_labels = merged["source_date"].dt.strftime("%b-%y")
        plt.figure(figsize=(6,4))
        plt.bar(x_labels, merged["diff_newmean_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("diff_newmean_pct")
        plt.title(f"{fname} %Δ newmean over time")
        plt.tight_layout()
        plt.show()

