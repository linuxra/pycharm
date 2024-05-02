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
    business_area: str = field(init=False)  # Add business area as a field

    def __post_init__(self):
        # Validate the yyyymm format
        self.validate_date_format()

        # Set the full file path using pathlib
        self.file_path = (
            self.base_path / f"{self.model_segment}_{self.yyyymm}{self.extension}"
        )

        # Determine business area based on model segment
        self.business_area = self.determine_business_area()

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


# Usage example:
try:
    config = Config(model_segment="sales", yyyymm="202305")
    print(f"The data file path is: {config.file_path}")
    print(f"Business Area: {config.business_area}")
except ValueError as e:
    print(e)
