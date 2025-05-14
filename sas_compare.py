import os
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt


def read_sas_dataframe(path: str) -> pd.DataFrame:
    """
    Read a .sas7bdat file into a pandas DataFrame.
    """
    return pd.read_sas(path, format="sas7bdat", encoding="utf-8")


def filter_actual(df: pd.DataFrame, grp_col: str = "grp") -> pd.DataFrame:
    """
    Filter the DataFrame to only rows where grp_col == 'actual'.
    """
    return df[df[grp_col] == "actual"].copy()


def compute_diff(
    old: pd.DataFrame,
    new: pd.DataFrame,
    key_cols: List[str],
    value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge old/new on key_cols, compute mean of each value_col, and
    calculate absolute and percent differences.

    Returns a DataFrame with one row per key and columns:
      - old_<col>_mean
      - new_<col>_mean
      - diff_<col>
      - diff_<col>_pct
    """
    # Identify which columns to compare
    if value_cols is None:
        exclude = set(key_cols + ["grp"])
        value_cols = [
            c
            for c in old.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(old[c])
        ]

    # Group & mean
    old_means = (
        old.groupby(key_cols)[value_cols].mean().add_prefix("old_").reset_index()
    )
    new_means = (
        new.groupby(key_cols)[value_cols].mean().add_prefix("new_").reset_index()
    )

    # Merge
    merged = pd.merge(old_means, new_means, on=key_cols, how="inner")

    # Compute diffs
    for col in value_cols:
        o = f"old_{col}"
        n = f"new_{col}"
        merged[f"diff_{col}"] = merged[n] - merged[o]
        # guard divide by zero
        merged[f"diff_{col}_pct"] = (
            merged[f"diff_{col}"] / merged[o].replace({0: pd.NA}) * 100
        )

    return merged


def plot_diff(
    df: pd.DataFrame, key_cols: List[str], diff_col: str, title: Optional[str] = None
):
    """
    Bar‐plot of a single diff column (absolute or percent) against the key.
    """
    x = df[key_cols].astype(str).agg("-".join, axis=1)
    y = df[diff_col]

    plt.figure(figsize=(8, 4))
    plt.bar(x, y)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(diff_col)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def compare_sas_directories(
    dir_old: str,
    dir_new: str,
    key_cols: List[str],
    grp_col: str = "grp",
    value_cols: Optional[List[str]] = None,
):
    """
    For each .sas7bdat file in dir_old, read the corresponding file
    from dir_new, filter grp='actual', compute mean differences
    and percent differences by key, and plot the results.

    Parameters
    ----------
    dir_old : str
        Path to the directory containing the “old” .sas7bdat files.
    dir_new : str
        Path to the directory containing the “new” .sas7bdat files.
    key_cols : List[str]
        Column name(s) to merge on.
    grp_col : str
        Name of the grouping column to filter on (default "grp").
    value_cols : Optional[List[str]]
        Specific numeric columns to compare. If None, all numeric
        columns except keys and grp_col will be used.
    """
    for fname in os.listdir(dir_old):
        if not fname.lower().endswith(".sas7bdat"):
            continue

        old_path = os.path.join(dir_old, fname)
        new_path = os.path.join(dir_new, fname)
        if not os.path.exists(new_path):
            print(f"Skipping {fname!r}: no match in new directory.")
            continue

        # Read and filter
        df_old = filter_actual(read_sas_dataframe(old_path), grp_col)
        df_new = filter_actual(read_sas_dataframe(new_path), grp_col)

        # Compute diffs
        diffs = compute_diff(df_old, df_new, key_cols, value_cols)
        print(f"\nDifferences for {fname}:")
        print(diffs)

        # Plot percent differences for each value column
        pct_cols = [
            c for c in diffs.columns if c.startswith("diff_") and c.endswith("_pct")
        ]
        for pct in pct_cols:
            plot_diff(diffs, key_cols, pct, title=f"{fname} – {pct}")
