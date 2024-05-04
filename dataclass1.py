from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Config:
    model_segment: str
    yyyymm: str
    base_path: Path = field(default_factory=lambda: Path("/path/to/data/"))
    extension: str = ".csv"
    file_path: Path = field(init=False)
    business_area: str = field(init=False)
    PSI_FILE_NAME: Path = field(init=False)  # PSI file name

    def __post_init__(self):
        # Validate the yyyymm format
        self.validate_date_format()

        # Set the full file path using pathlib
        self.file_path = (
            self.base_path / f"{self.model_segment}_{self.yyyymm}{self.extension}"
        )

        # Determine business area based on model segment
        self.business_area = self.determine_business_area()

        # Set the PSI file name
        self.PSI_FILE_NAME = self.generate_file_name("PSI")

    def validate_date_format(self):
        """Ensure yyyymm is a valid date string in the format YYYYMM."""
        try:
            datetime.strptime(self.yyyymm, "%Y%m")
        except ValueError:
            raise ValueError("yyyymm must be in YYYYMM format")

    def determine_business_area(self):
        """Map model segment to business area."""
        mapping = {
            "sales": "Sales and Marketing",
            "marketing": "Marketing Department",
            "finance": "Financial Operations",
            "hr": "Human Resources",
            "it": "Information Technology",
        }
        return mapping.get(self.model_segment, "Unknown Business Area")

    def generate_file_name(self, file_type):
        """Generate a file name based on the type, model segment, and date."""
        return (
            self.base_path
            / f"{self.model_segment}_{self.yyyymm}_{file_type}{self.extension}"
        )


# Usage example:
try:
    config = Config(model_segment="sales", yyyymm="202305")
    print(f"The data file path is: {config.file_path}")
    print(f"Business Area: {config.business_area}")
    print(f"PSI File Name: {config.PSI_FILE_NAME}")
except ValueError as e:
    print(e)
