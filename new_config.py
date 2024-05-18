from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Constant model ID
MODEL_ID = "mdl_102"


@dataclass
class Config:
    """
    Configuration class for generating file paths and names based on business area, date, and optional performance window.

    Attributes:
        bus_area (str): Business area.
        yyyymm (str): Year and month in YYYYMM format.
        perf (Optional[int]): Optional performance window (e.g., 6, 9, 12, 18, 24).
    """

    bus_area: str
    yyyymm: str
    perf: Optional[int] = None

    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    common_dir: Path = field(init=False)
    notebooks_dir: Path = field(init=False)
    utils_dir: Path = field(init=False)
    temp_dir: Path = field(init=False)
    dump_dir: Path = field(init=False)
    archive_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    scr_file_name: Path = field(init=False)
    perf_file_name: Optional[Path] = field(init=False, default=None)
    mad_file_name: Optional[Path] = field(init=False, default=None)
    sd_file_name: Optional[Path] = field(init=False, default=None)
    rev_file_name: Optional[Path] = field(init=False, default=None)
    rev_parquet_file_name: Optional[Path] = field(init=False, default=None)
    psi_file_name: Path = field(init=False)
    psi_metadata_file: Path = field(init=False)
    mer_metadata_file: Optional[Path] = field(init=False, default=None)
    log_file_name: Path = field(init=False)

    def __post_init__(self):
        """
        Post-initialization to set up directories and generate file names.
        """
        # Set base directory
        self.base_dir = Path(f"/users/modeling/{MODEL_ID}/malts/model_id")

        # Define directories
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.metadata_dir = self.base_dir / "metadata"
        self.common_dir = self.base_dir / "common"
        self.notebooks_dir = self.base_dir / "notebooks"
        self.utils_dir = self.base_dir / "utils"
        self.temp_dir = self.base_dir / "temp"
        self.dump_dir = self.base_dir / "dump"
        self.archive_dir = self.base_dir / "archive"
        self.logs_dir = self.base_dir / "logs"

        # Generate filenames
        self.scr_file_name = self._generate_scr_filename()
        self.psi_file_name = self._generate_output_filename("psi", always=True)
        self.psi_metadata_file = self._generate_metadata_filename("psi")

        if self.perf is not None:
            self.perf_file_name = self._generate_perf_filename()
            self.mad_file_name = self._generate_output_filename("mad")
            self.sd_file_name = self._generate_output_filename("sd")
            self.rev_file_name = self._generate_output_filename("rev")
            self.rev_parquet_file_name = self._generate_rev_parquet_filename()
            self.mer_metadata_file = self._generate_metadata_filename("mer")
        else:
            self.perf_file_name = None
            self.mad_file_name = None
            self.sd_file_name = None
            self.rev_file_name = None
            self.rev_parquet_file_name = None
            self.mer_metadata_file = None

        self.log_file_name = self._generate_log_filename()

    def _generate_scr_filename(self) -> Path:
        """
        Generate the filename for the scr file.

        Returns:
            Path: Full path to the scr file.
        """
        filename = f"{MODEL_ID}_{self.bus_area}_scr_{self.yyyymm}.parquet"
        return self.data_dir / filename

    def _generate_perf_filename(self) -> Path:
        """
        Generate the filename for the perf file.

        Returns:
            Path: Full path to the perf file.
        """
        filename = f"{MODEL_ID}_{self.bus_area}_perf_{self.perf}_{self.yyyymm}.parquet"
        return self.data_dir / filename

    def _generate_rev_parquet_filename(self) -> Path:
        """
        Generate the filename for the rev file with parquet extension.

        Returns:
            Path: Full path to the rev parquet file.
        """
        filename = f"{MODEL_ID}_{self.bus_area}_rev_{self.perf}_{self.yyyymm}.parquet"
        return self.data_dir / filename

    def _generate_output_filename(
        self, summary_type: str, always: bool = False
    ) -> Optional[Path]:
        """
        Generate the filename for the summary files (mad, sd, rev, psi).

        Args:
            summary_type (str): Type of summary file (mad, sd, rev, psi).
            always (bool): Flag to always generate the file (used for psi).

        Returns:
            Optional[Path]: Full path to the summary file or None if conditions are not met.
        """
        valid_types = ["mad", "sd", "rev", "psi"]
        if summary_type.lower() in valid_types:
            if always or self.perf is not None:
                if summary_type.lower() == "psi":
                    filename = f"{MODEL_ID}_{self.bus_area}_{summary_type}_summ_{self.yyyymm}.csv"
                else:
                    filename = f"{MODEL_ID}_{self.bus_area}_{summary_type}_summ_{self.perf}_{self.yyyymm}.csv"
                return self.output_dir / f"{summary_type.upper()}_summaries" / filename
            else:
                return None
        else:
            raise ValueError(
                f"Invalid summary type. Must be one of: {', '.join(valid_types)}"
            )

    def _generate_metadata_filename(self, metadata_type: str) -> Optional[Path]:
        """
        Generate the filename for the metadata files (psi, mer).

        Args:
            metadata_type (str): Type of metadata file (psi, mer).

        Returns:
            Optional[Path]: Full path to the metadata file or None if conditions are not met.
        """
        if metadata_type.lower() == "psi":
            filename = f"psi_{self.bus_area}_baseline.csv"
        elif metadata_type.lower() == "mer" and self.perf is not None:
            filename = f"mer_{self.bus_area}_{self.perf}_baseline.csv"
        else:
            return None
        return self.metadata_dir / filename

    def _generate_log_filename(self) -> Path:
        """
        Generate the filename for the log file.

        Returns:
            Path: Full path to the log file.
        """
        if self.perf is not None:
            filename = f"{MODEL_ID}_{self.bus_area}_{self._determine_file_type()}_{self.perf}_{self.yyyymm}.log"
        else:
            filename = f"{MODEL_ID}_{self.bus_area}_{self._determine_file_type()}_{self.yyyymm}.log"
        return self.logs_dir / filename

    def _determine_file_type(self) -> str:
        """
        Determine the file type based on the presence of various files.

        Returns:
            str: The file type.
        """
        if self.perf_file_name:
            return "perf"
        elif self.mad_file_name:
            return "mad"
        elif self.sd_file_name:
            return "sd"
        elif self.rev_file_name or self.rev_parquet_file_name:
            return "rev"
        elif self.psi_file_name:
            return "psi"
        else:
            return "scr"

    # def __repr__(self) -> str:
    #     """
    #     String representation of the Config class, printing each file path line by line.
    #
    #     Returns:
    #         str: Multiline string representation of the class.
    #     """
    #     attrs = [
    #         "bus_area",
    #         "yyyymm",
    #         "perf",
    #         "base_dir",
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
    #     return "Config:\n" + "\n".join(
    #         f"  {attr}: {getattr(self, attr)}"
    #         for attr in attrs
    #         if getattr(self, attr) is not None
    #     )

    def __repr__(self):
        fields = [
            f"{attr}: {getattr(self, attr)}" for attr in self.__dataclass_fields__
        ]
        return "Config(\n    " + ",\n    ".join(fields) + "\n)"


# Example usage
c1 = Config("finance", "201312", 6)
print(c1)

c2 = Config("finance", "201312")
print(c2)
# import os
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Optional, List
#
# # Constant model ID (can be configured via environment variable)
# MODEL_ID = os.getenv("MODEL_ID", "mdl_102")
# BASE_DIR = Path(os.getenv("BASE_DIR", f"/users/modeling/{MODEL_ID}/malts/model_id"))
#
#
# @dataclass
# class Config:
#     """
#     Configuration class for generating file paths and names based on business area, date, and optional performance window.
#
#     Attributes:
#         bus_area (str): Business area.
#         yyyymm (str): Year and month in YYYYMM format.
#         perf (Optional[int]): Optional performance window (e.g., 6, 9, 12, 18, 24).
#     """
#
#     bus_area: str
#     yyyymm: str
#     perf: Optional[int] = None
#
#     base_dir: Path = field(init=False, default=BASE_DIR)
#     data_dir: Path = field(init=False)
#     output_dir: Path = field(init=False)
#     metadata_dir: Path = field(init=False)
#     common_dir: Path = field(init=False)
#     notebooks_dir: Path = field(init=False)
#     utils_dir: Path = field(init=False)
#     temp_dir: Path = field(init=False)
#     dump_dir: Path = field(init=False)
#     archive_dir: Path = field(init=False)
#     logs_dir: Path = field(init=False)
#
#     scr_file_name: Path = field(init=False)
#     perf_file_name: Optional[Path] = field(init=False, default=None)
#     mad_file_name: Optional[Path] = field(init=False, default=None)
#     sd_file_name: Optional[Path] = field(init=False, default=None)
#     rev_file_name: Optional[Path] = field(init=False, default=None)
#     rev_parquet_file_name: Optional[Path] = field(init=False, default=None)
#     psi_file_name: Path = field(init=False)
#     psi_metadata_file: Path = field(init=False)
#     mer_metadata_file: Optional[Path] = field(init=False, default=None)
#     log_file_name: Path = field(init=False)
#
#     def __post_init__(self):
#         """
#         Post-initialization to set up directories and generate file names.
#         """
#         # Validate inputs
#         self._validate_inputs()
#
#         # Define directories
#         self.data_dir = self.base_dir / "data"
#         self.output_dir = self.base_dir / "output"
#         self.metadata_dir = self.base_dir / "metadata"
#         self.common_dir = self.base_dir / "common"
#         self.notebooks_dir = self.base_dir / "notebooks"
#         self.utils_dir = self.base_dir / "utils"
#         self.temp_dir = self.base_dir / "temp"
#         self.dump_dir = self.base_dir / "dump"
#         self.archive_dir = self.base_dir / "archive"
#         self.logs_dir = self.base_dir / "logs"
#
#         # Create directories if they don't exist
#         self._create_directories(
#             [
#                 self.data_dir,
#                 self.output_dir,
#                 self.metadata_dir,
#                 self.common_dir,
#                 self.notebooks_dir,
#                 self.utils_dir,
#                 self.temp_dir,
#                 self.dump_dir,
#                 self.archive_dir,
#                 self.logs_dir,
#             ]
#         )
#
#         # Generate filenames
#         self.scr_file_name = self._generate_scr_filename()
#         self.psi_file_name = self._generate_output_filename("psi", always=True)
#         self.psi_metadata_file = self._generate_metadata_filename("psi")
#
#         if self.perf is not None:
#             self.perf_file_name = self._generate_perf_filename()
#             self.mad_file_name = self._generate_output_filename("mad")
#             self.sd_file_name = self._generate_output_filename("sd")
#             self.rev_file_name = self._generate_output_filename("rev")
#             self.rev_parquet_file_name = self._generate_rev_parquet_filename()
#             self.mer_metadata_file = self._generate_metadata_filename("mer")
#         else:
#             self.perf_file_name = None
#             self.mad_file_name = None
#             self.sd_file_name = None
#             self.rev_file_name = None
#             self.rev_parquet_file_name = None
#             self.mer_metadata_file = None
#
#         self.log_file_name = self._generate_log_filename()
#
#     def _validate_inputs(self):
#         """
#         Validate the inputs to ensure they meet the expected formats and values.
#         """
#         if not self.bus_area:
#             raise ValueError("Business area must be provided.")
#
#         if not self.yyyymm or not self.yyyymm.isdigit() or len(self.yyyymm) != 6:
#             raise ValueError("yyyymm must be a string in the format YYYYMM.")
#
#         if self.perf is not None and self.perf not in {6, 9, 12, 18, 24}:
#             raise ValueError(
#                 "perf must be one of the following values: 6, 9, 12, 18, 24."
#             )
#
#     def _create_directories(self, directories: List[Path]):
#         """
#         Create directories if they don't exist.
#
#         Args:
#             directories (List[Path]): List of directory paths to create.
#         """
#         for directory in directories:
#             directory.mkdir(parents=True, exist_ok=True)
#
#     def _generate_scr_filename(self) -> Path:
#         """
#         Generate the filename for the scr file.
#
#         Returns:
#             Path: Full path to the scr file.
#         """
#         filename = f"{MODEL_ID}_{self.bus_area}_scr_{self.yyyymm}.parquet"
#         return self.data_dir / filename
#
#     def _generate_perf_filename(self) -> Path:
#         """
#         Generate the filename for the perf file.
#
#         Returns:
#             Path: Full path to the perf file.
#         """
#         filename = f"{MODEL_ID}_{self.bus_area}_perf_{self.perf}_{self.yyyymm}.parquet"
#         return self.data_dir / filename
#
#     def _generate_rev_parquet_filename(self) -> Path:
#         """
#         Generate the filename for the rev file with parquet extension.
#
#         Returns:
#             Path: Full path to the rev parquet file.
#         """
#         filename = f"{MODEL_ID}_{self.bus_area}_rev_{self.perf}_{self.yyyymm}.parquet"
#         return self.data_dir / filename
#
#     def _generate_output_filename(
#         self, summary_type: str, always: bool = False
#     ) -> Optional[Path]:
#         """
#         Generate the filename for the summary files (mad, sd, rev, psi).
#
#         Args:
#             summary_type (str): Type of summary file (mad, sd, rev, psi).
#             always (bool): Flag to always generate the file (used for psi).
#
#         Returns:
#             Optional[Path]: Full path to the summary file or None if conditions are not met.
#         """
#         valid_types = ["mad", "sd", "rev", "psi"]
#         if summary_type.lower() in valid_types:
#             if always or self.perf is not None:
#                 if summary_type.lower() == "psi":
#                     filename = f"{MODEL_ID}_{self.bus_area}_{summary_type}_summ_{self.yyyymm}.csv"
#                 else:
#                     filename = f"{MODEL_ID}_{self.bus_area}_{summary_type}_summ_{self.perf}_{self.yyyymm}.csv"
#                 return self.output_dir / f"{summary_type.upper()}_summaries" / filename
#             else:
#                 return None
#         else:
#             raise ValueError(
#                 f"Invalid summary type. Must be one of: {', '.join(valid_types)}"
#             )
#
#     def _generate_metadata_filename(self, metadata_type: str) -> Optional[Path]:
#         """
#         Generate the filename for the metadata files (psi, mer).
#
#         Args:
#             metadata_type (str): Type of metadata file (psi, mer).
#
#         Returns:
#             Optional[Path]: Full path to the metadata file or None if conditions are not met.
#         """
#         if metadata_type.lower() == "psi":
#             filename = f"psi_{self.bus_area}_baseline.csv"
#         elif metadata_type.lower() == "mer" and self.perf is not None:
#             filename = f"mer_{self.bus_area}_perf_baseline.csv"
#         else:
#             return None
#         return self.metadata_dir / filename
#
#     def _generate_log_filename(self) -> Path:
#         """
#         Generate the filename for the log file.
#
#         Returns:
#             Path: Full path to the log file.
#         """
#         if self.perf is not None:
#             filename = f"{MODEL_ID}_{self.bus_area}_{self._determine_file_type()}_{self.perf}_{self.yyyymm}.log"
#         else:
#             filename = f"{MODEL_ID}_{self.bus_area}_{self._determine_file_type()}_{self.yyyymm}.log"
#         return self.logs_dir / filename
#
#     def _determine_file_type(self) -> str:
#         """
#         Determine the file type based on the presence of various files.
#
#         Returns:
#             str: The file type.
#         """
#         if self.perf_file_name:
#             return "perf"
#         elif self.mad_file_name:
#             return "mad"
#         elif self.sd_file_name:
#             return "sd"
#         elif self.rev_file_name or self.rev_parquet_file_name:
#             return "rev"
#         elif self.psi_file_name:
#             return "psi"
#         else:
#             return "scr"
#
#     def __repr__(self):
#         fields = [f"{attr}: {getattr(self, attr)}" for attr in self.__dataclass_fields__]
#         return f"Config(\n    {',\n    '.join(fields)}\n)"
#
#
# # Example usage
# c1 = Config("finance", "201312", 6)
# print(c1)
#
# c2 = Config("finance", "201312")
# print(c2)
