# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Define the base Report class
class Report:
    def __init__(self, dataframe, title, **kwargs):
        self.dataframe = dataframe
        self.title = title
        self.properties = kwargs

    def generate_table(self):
        """This method generates a table using Matplotlib."""
        fig, ax =plt.subplots(1,1)
        ax.axis('tight')
        ax.axis('off')
        self.table = ax.table(cellText=self.dataframe.values,
                              colLabels=self.dataframe.columns,
                              cellLoc = 'center',
                              loc='center')
        self.fig = fig

    def apply_style(self):
        """This method applies common styling to the table."""
        self.table.auto_set_font_size(False)
        self.table.set_fontsize(self.properties.get('font_size', 10))

    def generate(self):
        """This method generates the report by creating and styling the table."""
        self.generate_table()
        self.apply_style()
        self.fig.suptitle(self.title)
        plt.show()


class DataFrameReport(Report):
    """This subclass of Report represents a report with a dataframe."""
    def __init__(self, dataframe, title, **kwargs):
        super().__init__(dataframe, title, **kwargs)

    def apply_style(self):
        """This method applies styling specific to DataFrameReport."""
        super().apply_style()
        # Apply additional or modified styling here


class MatplotlibReport(Report):
    """This subclass of Report represents a report with a matplotlib plot."""
    def __init__(self, dataframe, title, **kwargs):
        super().__init__(dataframe, title, **kwargs)

    def apply_style(self):
        """This method applies styling specific to MatplotlibReport."""
        super().apply_style()
        # Apply additional or modified styling here


class DataFramePlotReport(Report):
    """This subclass of Report represents a report with a dataframe and a plot."""
    def __init__(self, dataframe, title, **kwargs):
        super().__init__(dataframe, title, **kwargs)

    # If no specific styling is needed, no need to override apply_style method.
    # It will use the common styling from the base class.


class ReportFactory:
    """This class generates a report based on a given type."""
    @staticmethod
    def create_report(report_type, dataframe, title, **kwargs):
        if report_type == "dataframe":
            return DataFrameReport(dataframe, title, **kwargs)
        elif report_type == "matplotlib":
            return MatplotlibReport(dataframe, title, **kwargs)
        elif report_type == "dataframe_plot":
            return DataFramePlotReport(dataframe, title, **kwargs)
        else:
            raise ValueError(f"Invalid report_type: {report_type}")


# Create settings dictionary with report parameters
settings = {
    "rai": {
        "report_type": "dataframe",
        "dataframe": df,  # pandas dataframe
        "title": "Your RAI Report Title",
        "font_size": 10,
        # other properties to customize the look and feel of the report
    },
    # More settings...
}

# Generate report for each set of settings
for report_id, report_settings in settings.items():
    report = ReportFactory.create_report(report_settings['report_type'],
                                         report_settings['dataframe'],
                                         report_settings['title'],
                                         **report_settings)
    report.generate()  # generate the report
