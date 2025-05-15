import os
from typing import List
import pandas as pd


def read_sas_cols(path: str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_sas(path, format='sas7bdat', encoding='utf-8')
    return df[cols].copy()


def filter_actual(df: pd.DataFrame, grp_col: str = "grp") -> pd.DataFrame:
    return df[df[grp_col] == "actual"].copy()


def parse_source_date(df: pd.DataFrame, src_col: str = "source") -> pd.DataFrame:
    df = df.copy()
    df['source_date'] = pd.to_datetime(df[src_col], format='%b_%y')
    return df


def summarize_sas_diffs(
        dir_old: str,
        dir_new: str,
        key_cols: List[str],
        cols_to_read: List[str] = ["source", "grp", "newmean"],
        grp_col: str = "grp",
        src_col: str = "source"
) -> pd.DataFrame:
    """
    For each .sas7bdat in dir_old:
      - Read only cols_to_read, filter grp='actual'
      - Parse source→datetime
      - Merge on key_cols
      - Compute diff_newmean_pct
      - Sort by source_date, keep last N (48 if 'qtrly' else 144)
      - From that slice, pick the row with max diff_newmean_pct
      - Compute pct_range, min_diff_newmean_pct, max_diff_newmean_pct
      - Attach those plus filename to the summary row
    Returns a DataFrame with one summary row per file.
    """
    summaries = []

    for fname in os.listdir(dir_old):
        if not fname.lower().endswith(".sas7bdat"):
            continue

        old_path = os.path.join(dir_old, fname)
        new_path = os.path.join(dir_new, fname)
        if not os.path.exists(new_path):
            continue

        # --- read, filter, parse
        df_old = filter_actual(read_sas_cols(old_path, cols_to_read), grp_col)
        df_new = filter_actual(read_sas_cols(new_path, cols_to_read), grp_col)
        df_old = parse_source_date(df_old, src_col)
        df_new = parse_source_date(df_new, src_col)

        # --- merge & diff
        merged = pd.merge(
            df_old, df_new,
            on=key_cols,
            suffixes=("_old", "_new"),
            how="inner"
        )
        merged["diff_newmean"] = merged["newmean_new"] - merged["newmean_old"]
        merged["diff_newmean_pct"] = (
                merged["diff_newmean"]
                / merged["newmean_old"].replace({0: pd.NA})
                * 100
        )

        # --- slice to most recent N
        merged.sort_values("source_date", inplace=True)
        n = 48 if "qtrly" in fname.lower() else 144
        recent = merged.tail(n)
        if recent.empty:
            continue

        # --- compute min, max, range of pct diffs
        min_pct = recent["diff_newmean_pct"].min()
        max_pct = recent["diff_newmean_pct"].max()
        pct_range = max_pct - min_pct

        # --- pick the row with max diff_newmean_pct
        max_idx = recent["diff_newmean_pct"].idxmax()
        row = recent.loc[max_idx].copy()

        # --- attach summary metrics
        row["min_diff_newmean_pct"] = min_pct
        row["max_diff_newmean_pct"] = max_pct
        row["pct_range"] = pct_range
        row["filename"] = fname

        summaries.append(row)

    # assemble and return
    if summaries:
        return pd.DataFrame(summaries).reset_index(drop=True)
    else:
        return pd.DataFrame()


# Example usage:

if __name__ == "__main__":
    old_dir = "/data/sas/old_versions"
    new_dir = "/data/sas/new_versions"
    key_columns = ["source", "report_date"]

    summary_df = summarize_sas_diffs(
        dir_old=old_dir,
        dir_new=new_dir,
        key_cols=key_columns,
        cols_to_read=["source", "grp", "newmean"],
        grp_col="grp",
        src_col="source"
    )

    # Now display the combined one-row-per-file summary:
    print(summary_df)

