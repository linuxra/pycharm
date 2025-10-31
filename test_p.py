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
