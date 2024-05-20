import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import inspect

# Constants
MODEL_ID: str = os.getenv("MODEL_ID", "mdl_102")
BASE_DIR: Path = Path(
    os.getenv("BASE_DIR", f"/Users/rkaddanki/Documents/Python_Projects/pycharm/models/")
)
PERF_WINDOWS: tuple[int, ...] = (6, 9, 12, 18, 24)  # Possible performance windows


@dataclass
class Config:
    """Configuration for managing model-related file paths and names."""

    bus_area: str
    """The business area the model pertains to (e.g., "finance")."""
    yyyymm: str
    """The year and month in YYYYMM format (e.g., "202312")."""
    perf: Optional[Literal[6, 9, 12, 18, 24]] = None
    """Optional performance window for analysis."""

    # Base directory (initialized automatically)
    base_dir: Path = field(init=False, default=BASE_DIR)

    # --- Directories ---
    @property
    def data_dir(self) -> Path:
        """Directory for input data files."""
        return self.base_dir / "data"

    @property
    def output_dir(self) -> Path:
        """Directory for model output files."""
        return self.base_dir / "output"

    @property
    def metadata_dir(self) -> Path:
        """Directory for metadata files."""
        return self.base_dir / "metadata"

    @property
    def common_dir(self) -> Path:
        """Directory for common files."""
        return self.base_dir / "common"

    @property
    def notebooks_dir(self) -> Path:
        """Directory for Jupyter notebooks."""
        return self.base_dir / "notebooks"

    @property
    def utils_dir(self) -> Path:
        """Directory for utility scripts."""
        return self.base_dir / "utils"

    @property
    def temp_dir(self) -> Path:
        """Directory for temporary files."""
        return self.base_dir / "temp"

    @property
    def dump_dir(self) -> Path:
        """Directory for data dumps."""
        return self.base_dir / "dump"

    @property
    def archive_dir(self) -> Path:
        """Directory for archived files."""
        return self.base_dir / "archive"

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.base_dir / "logs"

    # --- File Names ---
    @property
    def scr_file_name(self) -> Path:
        """File name for the score (scr) file."""
        return self._generate_filename("scr", self.data_dir)

    @property
    def perf_file_name(self) -> Optional[Path]:
        """File name for the performance (perf) file (only if `perf` is specified)."""
        return self._generate_filename("perf", self.data_dir) if self.perf else None

    @property
    def mad_file_name(self) -> Optional[Path]:
        """File name for the Mean Absolute Deviation (MAD) summary file."""
        return (
            self._generate_filename("mad_summ", self.output_dir / "MAD_summaries")
            if self.perf
            else None
        )

    @property
    def sd_file_name(self) -> Optional[Path]:
        """File name for the Standard Deviation (SD) summary file."""
        return (
            self._generate_filename("sd_summ", self.output_dir / "SD_summaries")
            if self.perf
            else None
        )

    @property
    def rev_file_name(self) -> Optional[Path]:
        """File name for the revenue (rev) summary file."""
        return (
            self._generate_filename("rev_summ", self.output_dir / "REV_summaries")
            if self.perf
            else None
        )

    @property
    def rev_parquet_file_name(self) -> Optional[Path]:
        """File name for the revenue (rev) file in Parquet format."""
        return self._generate_filename("rev", self.data_dir) if self.perf else None

    @property
    def psi_file_name(self) -> Path:
        """File name for the Population Stability Index (PSI) summary file."""
        return self._generate_filename(
            "psi_summ", self.output_dir / "PSI_summaries", always=True
        )  # Always generate PSI

    @property
    def psi_metadata_file(self) -> Path:
        """File name for the PSI metadata file."""
        return self.metadata_dir / f"psi_{self.bus_area}_baseline.csv"

    @property
    def mer_metadata_file(self) -> Optional[Path]:
        """File name for the Marginal Effect at Reasonable (MER) metadata file."""
        return (
            self.metadata_dir / f"mer_{self.bus_area}_{self.perf}_baseline.csv"
            if self.perf
            else None
        )

    @property
    def log_file_name(self) -> Path:
        """File name for the log file."""
        file_type = (
            "perf"
            if self.perf_file_name
            else (
                "mad"
                if self.mad_file_name
                else (
                    "sd"
                    if self.sd_file_name
                    else ("rev" if self.rev_file_name else "psi")
                )
            )
        )

        # Generate the filename using the _generate_filename method
        log_file_path = self._generate_filename(file_type, self.logs_dir)

        # Modify the filename extension to '.log'
        return log_file_path.with_suffix(".log")

        # --- Methods ---

    def __post_init__(self):
        """Validate inputs and create directories after initialization."""
        self._validate_inputs()
        self._create_directories()

    def _validate_inputs(self):
        """Validate the provided configuration values."""
        if not self.bus_area:
            raise ValueError("Business area must be provided.")
        if not (self.yyyymm.isdigit() and len(self.yyyymm) == 6):
            raise ValueError("yyyymm must be a string in the format YYYYMM.")
        if self.perf and self.perf not in PERF_WINDOWS:
            raise ValueError(
                f"perf must be one of: {', '.join(map(str, PERF_WINDOWS))}."
            )

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.output_dir,
            self.metadata_dir,
            self.common_dir,
            self.notebooks_dir,
            self.utils_dir,
            self.temp_dir,
            self.dump_dir,
            self.archive_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _generate_filename(
        self, file_type: str, base_dir: Path, always: bool = False
    ) -> Path:
        """Generate a file name based on the file type and configuration."""
        filename = f"{MODEL_ID}_{self.bus_area}_{file_type}"
        if (
            not always and file_type != "scr"
        ):  # Only add perf if it's not 'scr' and not always
            filename += f"_{self.perf}" if self.perf else ""
        filename += f"_{self.yyyymm}.{'csv' if file_type == 'psi' else 'parquet'}"
        return base_dir / filename

    # def __repr__(self):
    #     """Provide a clear string representation of the configuration."""
    #     fields = [
    #         f"{attr}: {getattr(self, attr)}" for attr in self.__dataclass_fields__
    #     ]
    #     return "Config(\n    " + ",\n    ".join(fields) + "\n)"

    # def __repr__(self) -> str:
    #     """Provide a clear string representation of the configuration."""
    #     properties = [
    #         "data_dir",
    #         "output_dir",
    #         "metadata_dir",
    #         "common_dir",
    #         "notebooks_dir",
    #         "utils_dir",
    #         "temp_dir",
    #         "dump_dir",
    #         "archive_dir",
    #         "logs_dir",
    #         "scr_file_name",
    #         "perf_file_name",
    #         "mad_file_name",
    #         "sd_file_name",
    #         "rev_file_name",
    #         "rev_parquet_file_name",
    #         "psi_file_name",
    #         "psi_metadata_file",
    #         "mer_metadata_file",
    #         "log_file_name",
    #     ]
    #     fields = [f"{attr}: {getattr(self, attr)}" for attr in properties]
    #     return "Config(\n    " + ",\n    ".join(fields) + "\n)"

    # def __repr__(self) -> str:
    #     """Provide a clear string representation of the configuration, including properties."""
    #     fields = []
    #     for attr_name in self.__dataclass_fields__:
    #         attr_value = getattr(self, attr_name)
    #         if isinstance(attr_value, property):
    #             attr_value = attr_value.fget(self)  # Get the property's computed value
    #         fields.append(f"{attr_name}: {attr_value}")
    #
    #     return "Config(\n    " + ",\n    ".join(fields) + "\n)"
    def __repr__(self) -> str:
        """Provide a clear string representation of the configuration."""
        fields_str = []

        for key in dir(self):
            if key.startswith("_") or key in ["base_dir"]:
                continue
            value = getattr(self, key)

            if not callable(value) and not inspect.isroutine(value):
                fields_str.append(f"{key}: {value}")
        return f"Config(\n    " + ",\n    ".join(fields_str) + "\n)"

    # Example usage


# Example usage
c1 = Config("port", "201312", 6)
print(c1)

c2 = Config("finance", "201312")
print(c2)
