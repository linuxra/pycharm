import pandas as pd
from datetime import datetime


def load_data(bus_area, dt):
    """Loads parquet data based on the business area and date."""

    try:
        dt_obj = datetime.strptime(dt, "%Y%m")
    except ValueError:
        return None

    if bus_area == "student":
        # Quarter-end logic for students (same as before)
        is_quarter_end = dt_obj.month in [3, 6, 9, 12]
        if not is_quarter_end:
            return None

        start_month = (dt_obj.month - 3) if dt_obj.month > 3 else 1
        file_paths = [
            f"data/{dt_obj.year}{month:02d}.parquet"
            for month in range(start_month, dt_obj.month + 1)
        ]
        dfs = [pd.read_parquet(file_path) for file_path in file_paths]
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    else:
        # Standard monthly load for other business areas
        file_path = f"data/{dt_obj.year}{dt_obj.month:02d}.parquet"
        try:
            df = pd.read_parquet(file_path)
            return df
        except FileNotFoundError:  # Graceful handling if file is not found
            return None


# Example usage
business_areas = ["student", "teacher", "staff", "alumni"]
dt = "202311"  # Example date

for area in business_areas:
    loaded_data = load_data(area, dt)
    if loaded_data is not None:
        print(f"Loaded data for {area}:")
        print(loaded_data)
    else:
        print(
            f"No data loaded for {area} (not a quarter end for students or file not found)."
        )
