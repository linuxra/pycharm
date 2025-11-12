# model_metric_definition.py
from pydantic import BaseModel, Field, confloat, constr
from typing import Optional, List, Literal, Union
from datetime import date


class Range(BaseModel):
    """Numeric range for a given threshold."""

    GTE: Optional[confloat(ge=-1e6, le=1e6)] = Field(
        None, description="Greater than or equal to"
    )
    GT: Optional[confloat(ge=-1e6, le=1e6)] = Field(None, description="Greater than")
    LTE: Optional[confloat(ge=-1e6, le=1e6)] = Field(
        None, description="Less than or equal to"
    )
    LT: Optional[confloat(ge=-1e6, le=1e6)] = Field(None, description="Less than")


class Threshold(BaseModel):
    """Defines RYG threshold group (e.g., GREEN/YELLOW/RED) and its numeric ranges."""

    Key: Literal["RED", "YELLOW", "GREEN", "BLUE", "GRAY"] = Field(
        ..., description="Threshold bucket label"
    )
    Ranges: List[Range] = Field(
        ..., description="List of numeric range rules for this bucket"
    )


class ModelMetricDefinition(BaseModel):
    """Full definition of a model metric, including thresholds and metadata."""

    ModelId: int = Field(..., gt=0, description="Unique model identifier")
    ModelUseId: Optional[int] = Field(None, description="Optional model use identifier")
    MetricName: constr(min_length=1) = Field(
        ..., description="Human-readable metric name"
    )
    MetricId: constr(min_length=1) = Field(
        ..., description="System metric identifier (e.g., 'Performance', 'PSI')"
    )
    TestId: constr(min_length=1) = Field(
        ..., description="Unique test identifier for the metric"
    )
    Category: Literal["OUTCOME", "INPUT", "PROCESS", "QUALITY"] = Field(
        ..., description="Classification of the metric"
    )
    IsRanges: bool = Field(..., description="True if thresholds are numeric ranges")
    IsPercentage: bool = Field(..., description="True if metric is percentage-based")
    ThresholdType: Literal["RYG", "NUMERIC", "BOOLEAN", "TEXT"] = Field(
        ..., description="How thresholds are interpreted"
    )
    Thresholds: List[Threshold] = Field(
        ..., min_items=1, description="Threshold buckets"
    )
    ApplyFromDate: Optional[date] = Field(
        None, description="Effective start date for this definition"
    )
    Description: Optional[Union[dict, list]] = Field(
        None, description="Optional structured JSON metadata about this metric"
    )

    class Config:
        schema_extra = {
            "example": {
                "ModelId": 501319,
                "ModelUseId": 2001,
                "MetricName": "Day 2 Exceptions Aggregate",
                "MetricId": "Performance",
                "TestId": "501319_Day2ExceptionsAggregate_Performance",
                "Category": "OUTCOME",
                "IsRanges": True,
                "IsPercentage": True,
                "ThresholdType": "RYG",
                "Thresholds": [
                    {"Key": "GREEN", "Ranges": [{"GTE": -10.0, "LTE": 0.0}]},
                    {"Key": "YELLOW", "Ranges": [{"GTE": -20.0, "LT": -10.0}]},
                    {"Key": "RED", "Ranges": [{"GTE": -100.0, "LT": -20.0}]},
                ],
                "ApplyFromDate": "2023-08-01",
                "Description": {
                    "purpose": "Monitor day-2 exception rate performance",
                    "update_frequency": "Quarterly",
                    "owner": "Credit Risk Analytics - Cards",
                },
            }
        }


# loaders/excel_loader.py
import pandas as pd
import json
from pathlib import Path
from typing import Dict
from pydantic import ValidationError
from model_metric_definition import ModelMetricDefinition


def _coerce_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
    return val  # let Pydantic complain if it's something else


def _coerce_json(val, field_name, sheet_name):
    """
    For fields like Thresholds and Description that are stored as text JSON
    in Excel, convert them to real Python objects.
    """
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return None
        try:
            return json.loads(val)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Sheet '{sheet_name}': field '{field_name}' has invalid JSON.\n{e}"
            )
    return val


def load_model_definitions_from_excel(path: Path) -> Dict[str, ModelMetricDefinition]:
    """
    Reads one Excel file (one model), where each sheet is one metric definition.
    Returns {sheet_name: ModelMetricDefinition}
    """
    excel = pd.ExcelFile(path)
    out: Dict[str, ModelMetricDefinition] = {}

    for sheet_name in excel.sheet_names:
        # Expect two columns: VariableName | Value
        df = pd.read_excel(
            excel, sheet_name=sheet_name, header=None, names=["VariableName", "Value"]
        )
        df = df.dropna(subset=["VariableName"])  # ignore trailing blanks

        # Convert to dict of {VariableName: Value}
        raw_pairs = {
            str(row.VariableName).strip(): row.Value
            for row in df.itertuples(index=False)
        }

        # Coerce special fields
        if "Thresholds" in raw_pairs:
            raw_pairs["Thresholds"] = _coerce_json(
                raw_pairs["Thresholds"], "Thresholds", sheet_name
            )

        if "Description" in raw_pairs:
            raw_pairs["Description"] = _coerce_json(
                raw_pairs["Description"], "Description", sheet_name
            )

        for bool_field in ("IsRanges", "IsPercentage"):
            if bool_field in raw_pairs:
                raw_pairs[bool_field] = _coerce_bool(raw_pairs[bool_field])

        # Let Pydantic validate this sheet
        try:
            definition_obj = ModelMetricDefinition(**raw_pairs)
        except ValidationError as e:
            # You can also log-sheet-level context here
            print(f"Validation failed for sheet '{sheet_name}' in workbook {path.name}")
            print(e)
            raise

        out[sheet_name] = definition_obj

    return out


# loaders/yaml_writer.py
from pathlib import Path
import yaml
from model_metric_definition import ModelMetricDefinition


def write_definition_yaml(
    definition: ModelMetricDefinition, metric_name: str, out_dir: Path
) -> Path:
    """
    Dump a single metric definition to a YAML file like:
    {ModelId}_{MetricId}_definition.yaml
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = definition.ModelId
    metric_id = definition.MetricId  # e.g. "PSI", "MAD", etc.

    out_path = out_dir / f"{model_id}_{metric_id}_definition.yaml"

    # Convert Pydantic model to plain dict for YAML
    data_dict = definition.dict()

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data_dict,
            f,
            sort_keys=False,  # keep fields in natural order
            allow_unicode=True,  # keep unicode in owner/notes
        )

    return out_path


# generate_yaml_from_excel.py
from pathlib import Path
from loaders.excel_loader import load_model_definitions_from_excel
from loaders.yaml_writer import write_definition_yaml

EXCEL_PATH = Path("FICO08_CC_OD_Definitions.xlsx")
YAML_OUT_DIR = Path("out_definition_yaml")


def main():
    # Read and validate each metric definition in the workbook
    defs_by_metric = load_model_definitions_from_excel(EXCEL_PATH)

    # Write each one to YAML
    for metric_name, def_obj in defs_by_metric.items():
        out_path = write_definition_yaml(
            definition=def_obj, metric_name=metric_name, out_dir=YAML_OUT_DIR
        )
        print(f"✅ Wrote {out_path}")


if __name__ == "__main__":
    main()


# graphql_converter.py
from datetime import date
from typing import List, Dict, Any
from model_metric_definition import ModelMetricDefinition, Threshold, Range


def _format_bool(b: bool) -> str:
    # GraphQL wants true/false lowercase, unquoted
    return "true" if b else "false"


def _format_scalar(value: Any, treat_as_enum: bool = False) -> str:
    """
    value -> GraphQL literal.
    Rules:
      - enums like OUTCOME, RYG, GREEN must not be quoted
      - strings must be quoted
      - numbers stay as numbers
      - date -> "YYYY-MM-DD"
    """

    if value is None:
        return "null"  # we probably won't emit null fields but just in case

    # treat enums as bare words
    if treat_as_enum:
        return str(value)

    # booleans handled elsewhere
    if isinstance(value, bool):
        return _format_bool(value)

    # numeric types
    if isinstance(value, (int, float)):
        return str(value)

    # date -> quoted ISO
    if isinstance(value, date):
        return f'"{value.isoformat()}"'

    # string -> quoted
    if isinstance(value, str):
        # escape quotes inside string for safety
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'

    # fallback: dump as string quoted
    return f'"{str(value)}"'


def _format_range(rng: Range) -> str:
    """
    Range -> { GTE: -10.0, LTE: 0.0 }
    Only include keys that are not None.
    """
    parts = []
    if rng.GTE is not None:
        parts.append(f"GTE: {rng.GTE}")
    if rng.GT is not None:
        parts.append(f"GT: {rng.GT}")
    if rng.LTE is not None:
        parts.append(f"LTE: {rng.LTE}")
    if rng.LT is not None:
        parts.append(f"LT: {rng.LT}")
    inner = ", ".join(parts)
    return "{ " + inner + " }"


def _format_threshold(th: Threshold) -> str:
    """
    Threshold -> {
        Key: GREEN
        Ranges: [{ GTE: -10.0, LTE: 0.0 }]
    }
    """
    # Key is an enum-like token (GREEN / YELLOW / etc.) -> unquoted
    key_str = f"Key: {th.Key}"

    # Ranges is a list of range objects
    ranges_str = ", ".join(_format_range(r) for r in th.Ranges)
    ranges_str = f"Ranges: [{ranges_str}]"

    return "{ " + key_str + ", " + ranges_str + " }"


def definition_to_graphql_input(defn: ModelMetricDefinition) -> str:
    """
    Build the body of `input_data: { ... }` for CreateOmStgTestIdentification.
    We'll return just the {...} block so you can embed it in mutation.
    """

    fields = []

    # Required / common fields
    fields.append(f"ModelId: {defn.ModelId}")
    if defn.ModelUseId is not None:
        fields.append(f"ModelUseId: {defn.ModelUseId}")

    fields.append(f"MetricName: {_format_scalar(defn.MetricName)}")
    fields.append(f"MetricId: {_format_scalar(defn.MetricId)}")
    fields.append(f"TestId: {_format_scalar(defn.TestId)}")

    # Category is enum-like (OUTCOME, INPUT, etc.) -> unquoted
    fields.append(f"Category: {_format_scalar(defn.Category, treat_as_enum=True)}")

    # booleans
    fields.append(f"IsRanges: {_format_bool(defn.IsRanges)}")
    fields.append(f"IsPercentage: {_format_bool(defn.IsPercentage)}")

    # ThresholdType also looks like enum ("RYG", "NUMERIC"...)
    # If your API expects it unquoted like RYG, treat_as_enum=True.
    # If it expects "RYG", flip this.
    fields.append(
        f"ThresholdType: {_format_scalar(defn.ThresholdType, treat_as_enum=True)}"
    )

    # Thresholds block
    thresholds_formatted = ", ".join(_format_threshold(t) for t in defn.Thresholds)
    fields.append(f"Thresholds: [{thresholds_formatted}]")

    # ApplyFromDate
    if defn.ApplyFromDate is not None:
        fields.append(f"ApplyFromDate: {_format_scalar(defn.ApplyFromDate)}")

    # Description: This is optional structured JSON.
    # Question: does API even accept it? If yes, we need to serialize it properly.
    # If backend can't take arbitrary JSON here, comment this out.
    if defn.Description is not None:
        # We'll send Description as a quoted JSON string
        # because GraphQL input objects with arbitrary nested dynamic keys
        # may not be allowed unless schema supports it explicitly.
        import json

        desc_json = json.dumps(defn.Description)
        desc_json = desc_json.replace('"', '\\"')
        fields.append(f'Description: "{desc_json}"')

    inner = "\n    ".join(fields)
    return "{\n    " + inner + "\n}"


def build_full_mutation(defn: ModelMetricDefinition) -> str:
    """
    Wraps the input_data block into a full GraphQL mutation string.
    """
    input_block = definition_to_graphql_input(defn)

    mutation = (
        "mutation {\n"
        "  CreateOmStgTestIdentification(input_data: " + input_block + ")\n"
        "}"
    )

    return mutation


from pathlib import Path
from loaders.excel_loader import load_model_definitions_from_excel
from graphql_converter import build_full_mutation

excel_path = Path("FICO08_CC_OD_Definitions.xlsx")

defs_by_metric = load_model_definitions_from_excel(excel_path)

for metric_name, def_obj in defs_by_metric.items():
    print(f"--- Metric: {metric_name} ---")
    gql = build_full_mutation(def_obj)
    print(gql)
    print()


def build_full_mutation(defn: ModelMetricDefinition) -> str:
    """
    Wraps the input_data block into a full GraphQL mutation string.
    Includes return fields { success message errors { code description } data }
    """
    input_block = definition_to_graphql_input(defn)

    mutation = (
        "mutation {\n"
        "  CreateOmStgTestIdentification(input_data: " + input_block + ") {\n"
        "    success\n"
        "    message\n"
        "    errors { code description }\n"
        "    data\n"
        "  }\n"
        "}"
    )

    return mutation
[project]
name = "model_definitions"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pydantic==1.10.12", "pandas", "openpyxl", "pyyaml"]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
"""
Main driver script:
1. Reads one Excel file (one model) with multiple sheets (one metric each).
2. Validates with Pydantic.
3. Writes YAML files.
4. Generates GraphQL mutation strings.
"""

from pathlib import Path
from loaders.excel_loader import load_model_definitions_from_excel
from loaders.yaml_writer import write_definition_yaml
from graphql_converter import build_full_mutation


# === CONFIG ===
EXCEL_PATH = Path("FICO08_CC_OD_Definitions.xlsx")
YAML_OUT_DIR = Path("out_definition_yaml")
GRAPHQL_OUT_DIR = Path("out_mutations")


def main():
    YAML_OUT_DIR.mkdir(exist_ok=True)
    GRAPHQL_OUT_DIR.mkdir(exist_ok=True)

    # Step 1: Load and validate Excel definitions
    defs_by_metric = load_model_definitions_from_excel(EXCEL_PATH)

    # Step 2: Export YAML and GraphQL
    for metric_name, def_obj in defs_by_metric.items():
        # Write YAML
        yaml_path = write_definition_yaml(def_obj, YAML_OUT_DIR)
        print(f"✅ YAML written: {yaml_path}")

        # Write GraphQL mutation
        gql_str = build_full_mutation(def_obj)
        gql_path = GRAPHQL_OUT_DIR / f"{def_obj.ModelId}_{metric_name}_definition.graphql"
        gql_path.write_text(gql_str, encoding="utf-8")
        print(f"✅ GraphQL written: {gql_path}")


if __name__ == "__main__":
    main()


"""
model_metric_data_payload.py
----------------------------
Defines the Pydantic model for the Data Payload (metric-level JSON)
used in the OMR data-push process.

Python 3.10 compatible, Pydantic 1.10.12.
"""

from pydantic import BaseModel, Field, confloat
from datetime import date
from typing import Optional


class ModelMetricDataPayload(BaseModel):
    """
    Represents a single model–metric–month record to be pushed
    through GraphQL mutation.

    Each instance corresponds to ONE record (one .parquet file).
    """

    ModelId: int = Field(..., description="Unique model identifier")
    TestId: str = Field(..., description="Unique test identifier (ModelId_MetricId)")
    MetricId: str = Field(..., description="Metric type (e.g., PSI, SD, MAD)")
    ModelUseId: int = Field(..., description="Unique identifier from model registry")
    Result_date: date = Field(..., description="As-of date for the metric (1st of month)")
    Value: confloat(ge=-1e6, le=1e6) = Field(..., description="Metric numeric value")
    Info: Optional[str] = Field(
        None,
        description="Optional additional context (e.g., 'Auto Data Push 202505')"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"                # forbid unknown fields
        validate_assignment = True      # validate on assignment

    def to_dict(self) -> dict:
        """Return dict representation suitable for YAML or GraphQL serialization."""
        return self.dict(exclude_none=True)



"""
graphql_data_converter.py
--------------------------
Generates the GraphQL mutation text for ModelMetricDataPayload.

Maintains 1:1 structure with backend GraphQL schema.
"""

from code.model_metric_data_payload import ModelMetricDataPayload


def build_metric_data_mutation(p: ModelMetricDataPayload) -> str:
    """
    Construct GraphQL mutation text for a single payload.

    Parameters
    ----------
    p : ModelMetricDataPayload
        Validated Pydantic object.

    Returns
    -------
    str
        GraphQL mutation string ready for POST request.
    """
    return f"""
mutation {{
  CreateOmStgMetricData(input_data: {{
    ModelId: {p.ModelId}
    TestId: "{p.TestId}"
    MetricId: "{p.MetricId}"
    ModelUseId: {p.ModelUseId}
    Result_date: "{p.Result_date}"
    Value: {p.Value}
    Info: "{p.Info or ''}"
  }}) {{
    success
    message
    errors {{ code description }}
    data
  }}
}}
""".strip()


"""
constants.py
-------------
Holds global constants, base paths, and environment-specific configuration.
"""

from pathlib import Path

# Because we're inside code/common/, go up two levels to project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Core directories
METRICS_DIR = BASE_DIR / "data_push" / "metrics"
REGISTRY_PATH = BASE_DIR / "metadata" / "model_use_registry.parquet"
YAML_OUT_DIR = BASE_DIR / "data_push" / "out_data_yaml"
GRAPHQL_OUT_DIR = BASE_DIR / "data_push" / "out_data_mutations"
LOG_DIR = BASE_DIR / "logs"

# API Configuration (replace later with secure secrets)
ENDPOINT = "https://your-api-endpoint/graphql"
TOKEN = "YOUR_SECURE_TOKEN"

# Generic info text used in payload
DEFAULT_INFO = "Automated Data Push"


"""
date_utils.py
--------------
Provides month/quarter utility helpers for metric payload generation.
"""

from datetime import datetime
from typing import List


def quarter_end_yyyymm(year: int, quarter: int) -> str:
    """Return the ending YYYYMM for a given quarter."""
    month_map = {1: 3, 2: 6, 3: 9, 4: 12}
    return f"{year}{month_map[quarter]:02d}"


def last_n_months(end_yyyymm: str, n: int) -> List[str]:
    """
    Generate list of YYYYMM strings for previous n months
    including end_yyyymm.
    """
    y, m = int(end_yyyymm[:4]), int(end_yyyymm[4:6])
    months = []
    for _ in range(n):
        months.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return months


def yyyymm_to_date(yyyymm: str) -> str:
    """Convert YYYYMM → YYYY-MM-01 string (Result_date format)."""
    return f"{yyyymm[:4]}-{yyyymm[4:]}-01"
"""
utils.py
---------
Utility functions shared across scripts (logging, directories, etc.)
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_dir: Path, log_prefix: str) -> logging.Logger:
    """
    Create and return a logger that writes to a timestamped file.

    Example
    -------
    >>> logger = setup_logger(LOG_DIR, "data_push_501319_PSI_202505")
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_prefix}_{ts}.log"

    logger = logging.getLogger(log_prefix)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # also log to console
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"Logger initialized → {log_path}")
    return logger
"""
generate_and_submit_single.py
-----------------------------
Executes one GraphQL data push for a single
(ModelId, YYYYMM, BusArea, MetricType).
"""

import sys
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from code.model_metric_data_payload import ModelMetricDataPayload
from code.graphql_data_converter import build_metric_data_mutation
from code.common.constants import METRICS_DIR, REGISTRY_PATH, ENDPOINT, TOKEN, LOG_DIR, DEFAULT_INFO
from code.common.date_utils import yyyymm_to_date
from code.common.utils import setup_logger


def load_metric_value(model_id: int, metric: str, yyyymm: str):
    """
    Load metric value for given model, metric, and month.
    Expected file: data_push/metrics/<ModelId>/<Metric>/<Metric>_<YYYYMM>.parquet
    """
    file_path = METRICS_DIR / str(model_id) / metric / f"{metric}_{yyyymm}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_parquet(file_path)
    if df.empty:
        raise ValueError(f"Empty file: {file_path}")

    value = float(df["MetricValue"].iloc[0])
    business_area = df["BusinessArea"].iloc[0]
    return value, business_area


def lookup_model_use_id(model_id: int, metric: str, bus_area: str) -> int:
    """Fetch ModelUseId from registry parquet file."""
    df = pd.read_parquet(REGISTRY_PATH)
    row = df[
        (df["ModelId"] == model_id)
        & (df["MetricId"] == metric)
        & (df["BusinessArea"] == bus_area)
    ]
    if row.empty:
        raise ValueError(f"No registry entry for {model_id}-{metric}-{bus_area}")
    return int(row["ModelUseId"].iloc[0])


def push_payload(model_id: int, metric: str, bus_area: str, yyyymm: str):
    """Main function to push one metric payload."""
    logger = setup_logger(LOG_DIR, f"data_push_{model_id}_{metric}_{yyyymm}")
    logger.info(f"Starting push for {model_id}-{metric}-{yyyymm}")

    value, business_area = load_metric_value(model_id, metric, yyyymm)
    model_use_id = lookup_model_use_id(model_id, metric, bus_area)

    payload = ModelMetricDataPayload(
        ModelId=model_id,
        TestId=f"{model_id}_{metric}",
        MetricId=metric,
        ModelUseId=model_use_id,
        Result_date=yyyymm_to_date(yyyymm),
        Value=value,
        Info=f"{DEFAULT_INFO} {yyyymm}"
    )

    mutation = build_metric_data_mutation(payload)

    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    response = requests.post(ENDPOINT, json={"query": mutation}, headers=headers)

    if response.status_code != 200:
        logger.error(f"GraphQL error: {response.text}")
        raise RuntimeError(f"GraphQL error: {response.text}")

    logger.info(f"✅ Push complete for {metric} {yyyymm}")
    logger.info(response.json())


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python scripts/generate_and_submit_single.py <ModelId> <YYYYMM> <BusArea> <Metric>")
        sys.exit(1)

    model_id, yyyymm, bus_area, metric = sys.argv[1:]
    push_payload(int(model_id), metric, bus_area, yyyymm)


"""
write_metric_files_from_long_table.py
-------------------------------------
Splits a long-format metric DataFrame (or CSV) into individual Parquet files
following the directory convention:

    data_push/metrics/<ModelId>/<MetricId>/<MetricId>_<YYYYMM>.parquet

Each file contains one row of data corresponding to that unique combination.
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# add code directory to import path
sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))
from common.constants import METRICS_DIR


def write_metric_files(df: pd.DataFrame):
    """
    Save each unique (ModelId, MetricId, AsOfYYYYMM) row to its respective Parquet file.
    Expects columns: ModelId, BusinessArea, MetricId, AsOfYYYYMM, MetricValue
    """
    required_cols = {"ModelId", "BusinessArea", "MetricId", "AsOfYYYYMM", "MetricValue"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    for _, row in df.iterrows():
        model_id = int(row["ModelId"])
        metric = str(row["MetricId"]).upper()
        yyyymm = str(row["AsOfYYYYMM"])
        value = float(row["MetricValue"])

        model_dir = METRICS_DIR / str(model_id) / metric
        model_dir.mkdir(parents=True, exist_ok=True)
        file_path = model_dir / f"{metric}_{yyyymm}.parquet"

        row_df = pd.DataFrame([row])
        row_df.to_parquet(file_path, index=False)
        print(f"✅ Saved: {file_path}")


def main(input_path: Path):
    """
    Main entry — reads CSV or Parquet and calls write_metric_files().
    """
    ext = input_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Unsupported input file type. Use .csv or .parquet")

    print(f"📦 Loaded {len(df)} records from {input_path}")
    write_metric_files(df)
    print("🎉 All metric files written successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/write_metric_files_from_long_table.py <path_to_input_data>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    main(input_file)

from pathlib import Path
import pandas as pd
from code.model_metric_data_payload import ModelMetricDataPayload
from code.common.constants import METRICS_DIR, REGISTRY_PATH, DEFAULT_INFO
from code.common.date_utils import yyyymm_to_date

def load_metric_value(model_id: int, metric: str, yyyymm: str):
    file_path = METRICS_DIR / str(model_id) / metric / f"{metric}_{yyyymm}.parquet"
    df = pd.read_parquet(file_path)
    value = float(df["MetricValue"].iloc[0])
    business_area = df["BusinessArea"].iloc[0]
    return value, business_area

def lookup_model_use_id(model_id: int, metric: str, bus_area: str) -> int:
    df = pd.read_parquet(REGISTRY_PATH)
    row = df[
        (df["ModelId"] == model_id)
        & (df["MetricId"] == metric)
        & (df["BusinessArea"] == bus_area)
    ]
    if row.empty:
        raise ValueError(f"No registry entry for {model_id}-{metric}-{bus_area}")
    return int(row["ModelUseId"].iloc[0])

def create_payload(model_id: int, metric: str, bus_area: str, yyyymm: str) -> ModelMetricDataPayload:
    """Build and return validated payload object without sending."""
    value, business_area = load_metric_value(model_id, metric, yyyymm)
    model_use_id = lookup_model_use_id(model_id, metric, bus_area)
    return ModelMetricDataPayload(
        ModelId=model_id,
        TestId=f"{model_id}_{metric}_{bus_area or 'NA'}",
        MetricId=metric,
        ModelUseId=model_use_id,
        Result_date=yyyymm_to_date(yyyymm),
        Value=value,
        Info=f"{DEFAULT_INFO} {yyyymm}",
    )
"""
push_payload.py
---------------
Takes a payload object or JSON and sends it to GraphQL.
"""

import sys
import requests
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))

from payload_builder import create_payload
from graphql_data_converter import build_metric_data_mutation
from common.constants import ENDPOINT, TOKEN, LOG_DIR
from common.utils import setup_logger

def push_payload(model_id: int, metric: str, bus_area: str, yyyymm: str):
    logger = setup_logger(LOG_DIR, f"data_push_{model_id}_{metric}_{yyyymm}")
    payload = create_payload(model_id, metric, bus_area, yyyymm)
    mutation = build_metric_data_mutation(payload)

    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    response = requests.post(ENDPOINT, json={"query": mutation}, headers=headers)
    if response.status_code != 200:
        logger.error(f"GraphQL error: {response.text}")
        raise RuntimeError(response.text)
    logger.info(f"✅ Push complete for {payload.TestId} {yyyymm}")
    logger.info(response.json())

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python scripts/push_payload.py <ModelId> <YYYYMM> <BusArea> <Metric>")
        sys.exit(1)
    model_id, yyyymm, bus_area, metric = sys.argv[1:]
    push_payload(int(model_id), metric, bus_area, yyyymm)


"""
payload_builder.py
------------------
Creates and validates metric data payloads for a single
(ModelId, MetricId, BusinessArea, YYYYMM), builds GraphQL mutations,
and saves all artifacts (.yaml, .json, .graphql) for later submission.

Flow:
1. Read metric value from:
       data_push/metrics/<ModelId>/<MetricId>/<MetricId>_<YYYYMM>.parquet
2. Look up ModelUseId & TestId from model_use_registry.parquet
3. Build ModelMetricDataPayload (Pydantic validated)
4. Generate GraphQL mutation text
5. Save to:
       data_push/out_data_yaml/<ModelId>_<MetricId>_<YYYYMM>.yaml
       data_push/out_data_yaml/<ModelId>_<MetricId>_<YYYYMM>.json
       data_push/out_data_mutations/<ModelId>_<MetricId>_<YYYYMM>.graphql
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from model_metric_data_payload import ModelMetricDataPayload
from graphql_data_converter import build_metric_data_mutation
from common.constants import (
    METRICS_DIR,
    REGISTRY_PATH,
    DEFAULT_INFO,
    YAML_OUT_DIR,
    GRAPHQL_OUT_DIR,
)
from common.date_utils import yyyymm_to_date


def load_metric_value(model_id: int, metric_id: str, yyyymm: str):
    """
    Load metric value for given model, metric, and month.

    Expects file:
        data_push/metrics/<ModelId>/<MetricId>/<MetricId>_<YYYYMM>.parquet

    Parquet must contain columns:
        - MetricValue
        - BusinessArea
    """
    file_path = METRICS_DIR / str(model_id) / metric_id / f"{metric_id}_{yyyymm}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing metric file: {file_path}")

    df = pd.read_parquet(file_path)
    if df.empty:
        raise ValueError(f"Empty metric file: {file_path}")

    value = float(df["MetricValue"].iloc[0])
    business_area = str(df["BusinessArea"].iloc[0]).upper()
    return value, business_area


def lookup_registry(model_id: int, metric_id: str, bus_area: str):
    """
    Look up (ModelUseId, TestId) from registry.

    Registry schema must contain:
        ModelId, BusinessArea, MetricId, TestId, ModelUseId

    BusinessArea + MetricId comparisons are case-insensitive normalized to upper.
    """
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    df = pd.read_parquet(REGISTRY_PATH)

    metric_id_u = metric_id.upper()
    bus_area_u = (bus_area or "NA").upper()

    row = df[
        (df["ModelId"] == model_id)
        & (df["MetricId"].str.upper() == metric_id_u)
        & (df["BusinessArea"].str.upper() == bus_area_u)
    ]

    if row.empty:
        raise ValueError(
            f"No registry entry for ModelId={model_id}, MetricId={metric_id_u}, "
            f"BusinessArea={bus_area_u}"
        )

    model_use_id = int(row["ModelUseId"].iloc[0])
    test_id = str(row["TestId"].iloc[0])

    return model_use_id, test_id


def create_payload(model_id: int, metric_id: str, bus_area: str, yyyymm: str) -> ModelMetricDataPayload:
    """
    Build a validated ModelMetricDataPayload for one model-metric-month.

    TestId is NOT recomputed here; it is read from the registry to ensure
    it matches the definition-time convention:
        {ModelId}_{MetricNameNoSpaces}_{BusinessArea|NA}
    """
    value, inferred_bus_area = load_metric_value(model_id, metric_id, yyyymm)

    # Prefer explicit bus_area parameter, but ensure consistency with file.
    bus_area_u = (bus_area or inferred_bus_area or "NA").upper()

    model_use_id, test_id = lookup_registry(model_id, metric_id, bus_area_u)

    payload = ModelMetricDataPayload(
        ModelId=model_id,
        TestId=test_id,
        MetricId=metric_id.upper(),
        ModelUseId=model_use_id,
        Result_date=yyyymm_to_date(yyyymm),
        Value=value,
        Info=f"{DEFAULT_INFO} {yyyymm}",
    )
    return payload


def save_payload_files(payload: ModelMetricDataPayload, model_id: int, metric_id: str, yyyymm: str) -> Path:
    """
    Save payload as:
      - YAML
      - JSON
      - GraphQL (.graphql)

    Returns
    -------
    Path
        Path to the .graphql file (for the push step to use).
    """
    YAML_OUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHQL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_id_u = metric_id.upper()
    base_name = f"{model_id}_{metric_id_u}_{yyyymm}"

    yaml_path = YAML_OUT_DIR / f"{base_name}.yaml"
    json_path = YAML_OUT_DIR / f"{base_name}.json"
    graphql_path = GRAPHQL_OUT_DIR / f"{base_name}.graphql"

    data = payload.dict(exclude_none=True)

    # YAML
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.safe_dump(data, yf, sort_keys=False)

    # JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, indent=2)

    # GraphQL
    mutation = build_metric_data_mutation(payload)
    graphql_path.write_text(mutation, encoding="utf-8")

    print(f"✅ Saved payload artifacts for {base_name}:")
    print(f"   - {yaml_path}")
    print(f"   - {json_path}")
    print(f"   - {graphql_path}")

    return graphql_path


def create_and_save_all(model_id: int, metric_id: str, bus_area: str, yyyymm: str) -> Path:
    """
    End-to-end:
      1. Build payload (from metrics + registry)
      2. Save YAML/JSON/GraphQL
      3. Return GraphQL file path.
    """
    payload = create_payload(model_id, metric_id, bus_area, yyyymm)
    return save_payload_files(payload, model_id, metric_id, yyyymm)


if __name__ == "__main__":
    # Allow direct CLI usage:
    # python -m code.payload_builder <ModelId> <MetricId> <BusinessArea> <YYYYMM>
    import sys

    if len(sys.argv) != 5:
        print("Usage: python -m code.payload_builder <ModelId> <MetricId> <BusinessArea> <YYYYMM>")
        sys.exit(1)

    model_id_arg = int(sys.argv[1])
    metric_id_arg = sys.argv[2]
    bus_area_arg = sys.argv[3]
    yyyymm_arg = sys.argv[4]

    create_and_save_all(model_id_arg, metric_id_arg, bus_area_arg, yyyymm_arg)


"""
push_payload.py
---------------
Reads a saved YAML or JSON payload and sends it via GraphQL.
"""

import sys
import requests
import yaml
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))

from model_metric_data_payload import ModelMetricDataPayload
from graphql_data_converter import build_metric_data_mutation
from common.constants import ENDPOINT, TOKEN, LOG_DIR
from common.utils import setup_logger


def load_payload(file_path: Path) -> ModelMetricDataPayload:
    """Load YAML or JSON payload and validate."""
    ext = file_path.suffix.lower()
    if ext == ".yaml":
        data = yaml.safe_load(open(file_path))
    elif ext == ".json":
        data = json.load(open(file_path))
    else:
        raise ValueError("Unsupported file type. Use .yaml or .json")

    payload = ModelMetricDataPayload(**data)
    return payload


def push_payload_from_file(file_path: Path):
    """Push a single payload file to GraphQL."""
    payload = load_payload(file_path)
    logger = setup_logger(LOG_DIR, f"push_{payload.TestId}")
    mutation = build_metric_data_mutation(payload)

    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    response = requests.post(ENDPOINT, json={"query": mutation}, headers=headers)

    if response.status_code != 200:
        logger.error(f"❌ GraphQL error: {response.text}")
        raise RuntimeError(response.text)

    logger.info(f"✅ Push complete for {payload.TestId}")
    logger.info(response.json())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/push_payload.py <path_to_payload.yaml|json>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    push_payload_from_file(file_path)



"""
Main driver utilities for definition generation.

Changes:
- Exposes a function that takes (model_id, business_area, metric_id)
- Reads from input/<model_id>_<business_area>_<metric_id>_def.xlsx
- Validates with Pydantic (via your existing loader)
- Writes YAML files and GraphQL mutation strings
- Includes a batch helper to loop over multiple (model_id, business_area, metric_id)

Requires:
- loaders.excel_loader.load_model_definitions_from_excel
- loaders.yaml_writer.write_definition_yaml
- graphql_converter.build_full_mutation
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

from loaders.excel_loader import load_model_definitions_from_excel
from loaders.yaml_writer import write_definition_yaml
from graphql_converter import build_full_mutation


# === DEFAULT PATHS ===
INPUT_DIR = Path("input")
YAML_OUT_DIR = Path("out_definition_yaml")
GRAPHQL_OUT_DIR = Path("out_mutations")


def generate_definition_artifacts(
    model_id: str,
    business_area: str,
    metric_id: str,
    input_dir: Path = INPUT_DIR,
    yaml_out_dir: Path = YAML_OUT_DIR,
    graphql_out_dir: Path = GRAPHQL_OUT_DIR,
) -> Dict[str, Path]:
    """
    Generate YAML + GraphQL artifacts for a single (model_id, business_area, metric_id).

    It expects an Excel file at:
        input_dir / f"{model_id}_{business_area}_{metric_id}_def.xlsx"

    The Excel may contain one or multiple sheets (one metric per sheet). If multiple
    are present, this function selects the sheet/definition that matches `metric_id`.
    If only one is present, that one is used.

    Parameters
    ----------
    model_id : str
        The model identifier (e.g., "FICO08").
    business_area : str
        The business area (e.g., "CC").
    metric_id : str
        The metric identifier (e.g., "PSI").
    input_dir : Path, optional
        Directory containing input Excel files. Defaults to "input/".
    yaml_out_dir : Path, optional
        Directory to write YAML outputs. Defaults to "out_definition_yaml/".
    graphql_out_dir : Path, optional
        Directory to write GraphQL outputs. Defaults to "out_mutations/".

    Returns
    -------
    Dict[str, Path]
        Dictionary with keys:
            - "excel_path": Path to the source Excel
            - "yaml_path": Path to the written YAML
            - "graphql_path": Path to the written GraphQL mutation

    Raises
    ------
    FileNotFoundError
        If the expected Excel file does not exist.
    ValueError
        If `metric_id` is not found among loaded definitions (when multiple present).
    """
    # Ensure output directories exist
    yaml_out_dir.mkdir(parents=True, exist_ok=True)
    graphql_out_dir.mkdir(parents=True, exist_ok=True)

    # Build expected Excel path
    excel_filename = f"{model_id}_{business_area}_{metric_id}_def.xlsx"
    excel_path = input_dir / excel_filename

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Input Excel not found: {excel_path}. "
            f"Expected pattern: <model>_<business>_<metric>_def.xlsx"
        )

    # Load and validate (returns a mapping: metric_name -> pydantic model object)
    defs_by_metric = load_model_definitions_from_excel(excel_path)

    # Select the right metric definition
    if len(defs_by_metric) == 0:
        raise ValueError(f"No definitions found in Excel: {excel_path}")

    if metric_id in defs_by_metric:
        def_obj = defs_by_metric[metric_id]
        selected_metric = metric_id
    elif len(defs_by_metric) == 1:
        # If only one sheet/metric, use it even if the name doesn't match exactly
        selected_metric, def_obj = next(iter(defs_by_metric.items()))
    else:
        available = ", ".join(defs_by_metric.keys())
        raise ValueError(
            f"Metric '{metric_id}' not found in {excel_path.name}. "
            f"Available metrics: {available}"
        )

    # Write YAML
    yaml_path = write_definition_yaml(def_obj, yaml_out_dir)
    print(f"✅ YAML written: {yaml_path}")

    # Build and write GraphQL
    gql_str = build_full_mutation(def_obj)
    gql_filename = f"{def_obj.ModelId}_{selected_metric}_definition.graphql"
    graphql_path = graphql_out_dir / gql_filename
    graphql_path.write_text(gql_str, encoding="utf-8")
    print(f"✅ GraphQL written: {graphql_path}")

    return {
        "excel_path": excel_path,
        "yaml_path": yaml_path,
        "graphql_path": graphql_path,
    }


def run_batch(
    items: Iterable[Tuple[str, str, str]],
    input_dir: Path = INPUT_DIR,
    yaml_out_dir: Path = YAML_OUT_DIR,
    graphql_out_dir: Path = GRAPHQL_OUT_DIR,
) -> Dict[Tuple[str, str, str], Dict[str, Path]]:
    """
    Run `generate_definition_artifacts` for a list of (model_id, business_area, metric_id).

    Parameters
    ----------
    items : Iterable[Tuple[str, str, str]]
        Sequence of triples like: [("FICO08", "CC", "PSI"), ("FICO08", "CC", "SD"), ...]
    input_dir : Path, optional
        Directory containing the input Excel files.
    yaml_out_dir : Path, optional
        Directory for YAML outputs.
    graphql_out_dir : Path, optional
        Directory for GraphQL outputs.

    Returns
    -------
    Dict[Tuple[str, str, str], Dict[str, Path]]
        Mapping of the input triple to the returned paths dictionary.
    """
    results: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    for model_id, business_area, metric_id in items:
        print(f"\n--- Processing {model_id}/{business_area}/{metric_id} ---")
        artifacts = generate_definition_artifacts(
            model_id=model_id,
            business_area=business_area,
            metric_id=metric_id,
            input_dir=input_dir,
            yaml_out_dir=yaml_out_dir,
            graphql_out_dir=graphql_out_dir,
        )
        results[(model_id, business_area, metric_id)] = artifacts
    return results


if __name__ == "__main__":
    # Example usage: loop a list of params
    triplets = [
        ("FICO08", "CC", "PSI"),
        ("FICO08", "CC", "SD"),
        ("FICO08", "CC", "MAD"),
    ]
    run_batch(triplets)



from pathlib import Path
import sys
import pandas as pd

# add code/ to path to reuse constants
sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))
from common.constants import REGISTRY_PATH, BASE_DIR

EXCEL_SRC = BASE_DIR / "metadata" / "model_use_registry.xlsx"

REQUIRED_COLS = {
    "ModelId", "BusinessArea", "MetricId", "MetricName",
    "TestId", "ModelUseId", "Category", "ThresholdType"
}

def validate_test_id(row) -> bool:
    expected = f"{int(row['ModelId'])}_{str(row['MetricName']).replace(' ', '')}_{(row['BusinessArea'] or 'NA')}"
    return str(row["TestId"]) == expected

def main():
    if not EXCEL_SRC.exists():
        raise FileNotFoundError(f"Missing Excel registry: {EXCEL_SRC}")

    df = pd.read_excel(EXCEL_SRC)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing columns: {missing}")

    # normalize
    df["BusinessArea"] = df["BusinessArea"].fillna("NA").str.upper()
    df["MetricId"] = df["MetricId"].str.upper()

    # validate TestId format
    bad = df[~df.apply(validate_test_id, axis=1)]
    if not bad.empty:
        cols = ["ModelId", "MetricName", "BusinessArea", "TestId"]
        raise ValueError(f"Invalid TestId rows:\n{bad[cols]}")

    # drop dupes by (ModelId, MetricId, BusinessArea), keep last edited row
    df = df.drop_duplicates(subset=["ModelId", "MetricId", "BusinessArea"], keep="last")

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REGISTRY_PATH, index=False)
    print(f"✅ Synced registry → {REGISTRY_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()

