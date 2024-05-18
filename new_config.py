from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MODEL_ID = "mdl_102"  # This is the constant model ID


@dataclass
class Config:
    bus_area: str
    yyyymm: str
    perf: Optional[int] = None

    def __post_init__(self):
        if self.perf is not None:
            self.base_dir = Path(
                f"/users/modeling/{MODEL_ID}/malts/model_id/perf_{self.perf}"
            )
        else:
            self.base_dir = Path(f"/users/modeling/{MODEL_ID}/malts/model_id")

        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.metadata_dir = self.base_dir / "metadata"

        # Generate filenames upon initialization
        self.scr_file_name = self._generate_scr_filename()
        self.psi_file_name = self._generate_output_filename("psi", always=True)
        self.psi_metadata_file = self._generate_metadata_filename("psi")

        if self.perf is not None:
            self.perf_file_name = self._generate_perf_filename()
            self.mad_file_name = self._generate_output_filename("mad")
            self.sd_file_name = self._generate_output_filename("sd")
            self.rev_file_name = self._generate_output_filename("rev")
            self.mer_metadata_file = self._generate_metadata_filename("mer")
        else:
            self.perf_file_name = None
            self.mad_file_name = None
            self.sd_file_name = None
            self.rev_file_name = None
            self.mer_metadata_file = None

    def _generate_scr_filename(self):
        filename = f"{MODEL_ID}_{self.bus_area}_scr_{self.yyyymm}.parquet"
        return self.data_dir / filename

    def _generate_perf_filename(self):
        filename = f"{MODEL_ID}_{self.bus_area}_perf_{self.perf}_{self.yyyymm}.parquet"
        return self.data_dir / filename

    def _generate_output_filename(self, summary_type: str, always: bool = False):
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

    def _generate_metadata_filename(self, metadata_type: str):
        if metadata_type.lower() == "psi":
            filename = f"psi_{self.bus_area}_baseline.csv"
        elif metadata_type.lower() == "mer" and self.perf is not None:
            filename = f"mer_{self.bus_area}_perf_baseline.csv"
        else:
            return None
        return self.metadata_dir / filename


# Example usage
c1 = Config("finance", "201312", 6)
print(c1.scr_file_name)  # For scr file
print(c1.perf_file_name)  # For perf file
print(c1.mad_file_name)  # For MAD summary file
print(c1.sd_file_name)  # For SD summary file
print(c1.rev_file_name)  # For REV summary file
print(c1.psi_file_name)  # For PSI summary file
print(c1.psi_metadata_file)  # For PSI metadata file
print(c1.mer_metadata_file)  # For MER metadata file

c2 = Config("finance", "201312")
print(c2.scr_file_name)  # For scr file
print(c2.perf_file_name)  # None, as perf is not specified
print(c2.mad_file_name)  # None, as perf is not specified
print(c2.sd_file_name)  # None, as perf is not specified
print(c2.rev_file_name)  # None, as perf is not specified
print(c2.psi_file_name)  # For PSI summary file
print(c2.psi_metadata_file)  # For PSI metadata file
print(c2.mer_metadata_file)  # None, as perf is not specified
