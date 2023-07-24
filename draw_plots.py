import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

class MyClass:
    def __init__(self, title):
        self.title = title

    def draw_plots(self, df):
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 2, figure=fig)

        # Define the location of each subplot using the GridSpec
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

        # Line plot in the third subplot
        sns.lineplot(data=df, x=df.columns[1], y=df.columns[2], hue=df.columns[0], ax=axs[2])

        # Stacked bar plot in the first subplot
        df_pivot = df.pivot(index=df.columns[1], columns=df.columns[0], values=df.columns[2])
        df_pivot.plot(kind='bar', stacked=True, ax=axs[0])

        # Define a custom colormap
        c_map = LinearSegmentedColormap.from_list('mycmap', ['yellow', 'green'])

        # Heatmap in the second subplot
        sns.heatmap(df_pivot, cmap=c_map, ax=axs[1])

        # Set the title of the figure
        fig.suptitle(self.title)

        plt.tight_layout()
        plt.show()

# Assuming you have a dataframe df
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6, 7, 8, 9, 6, 7, 8, 9]
})

obj = MyClass(title='My Awesome Plots')
obj.draw_plots(df)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

class MyClass:
    def __init__(self, title):
        self.title = title

    def draw_plots(self, df):
        groups = df[df.columns[0]].unique()
        fig = plt.figure(figsize=(10, 15))
        gs = GridSpec(len(groups), 2, figure=fig)

        for i, group in enumerate(groups):
            df_group = df[df[df.columns[0]] == group]

            # Line plot in the first column for each group
            ax = fig.add_subplot(gs[i, 0])
            sns.lineplot(data=df_group, x=df_group.columns[1], y=df_group.columns[2], ax=ax)
            ax.set_title(f'Line Plot for Group {group}')

            # Bar plot in the second column for each group
            ax = fig.add_subplot(gs[i, 1])
            sns.barplot(data=df_group, x=df_group.columns[1], y=df_group.columns[2], ax=ax)
            ax.set_title(f'Bar Plot for Group {group}')

        # Set the title of the figure
        fig.suptitle(self.title)

        plt.tight_layout()
        plt.show()

# Assuming you have a dataframe df
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6, 7, 8, 9, 6, 7, 8, 9]
})

obj = MyClass(title='My Awesome Plots')
obj.draw_plots(df)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

class MyClass:
    def __init__(self, title):
        self.title = title

    def draw_plots(self, df):
        groups = df[df.columns[0]].unique()
        fig = plt.figure(figsize=(10, 5 * len(groups)))
        gs = GridSpec(len(groups), 1, figure=fig)

        for i, group in enumerate(groups):
            df_group = df[df[df.columns[0]] == group]

            # Bubble plot for each group
            ax = fig.add_subplot(gs[i, 0])
            scatter = ax.scatter(x=df_group[df_group.columns[1]], y=df_group[df_group.columns[2]], s=df_group[df_group.columns[2]]*10, alpha=0.5)
            ax.set_title(f'Bubble Plot for Group {group}')
            ax.set_xlabel(df_group.columns[1])
            ax.set_ylabel(df_group.columns[2])

        # Set the title of the figure
        fig.suptitle(self.title)

        plt.tight_layout()
        plt.show()

# Assuming you have a dataframe df
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6, 7, 8, 9, 6, 7, 8, 9]
})

obj = MyClass(title='My Bubble Plots')
obj.draw_plots(df)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler

class MyClass:
    def __init__(self, title):
        self.title = title

    def draw_plots(self, df):
        groups = df[df.columns[0]].unique()
        fig = plt.figure(figsize=(10, 5 * len(groups)))
        gs = GridSpec(len(groups), 1, figure=fig)

        scaler = MinMaxScaler()

        for i, group in enumerate(groups):
            df_group = df[df[df.columns[0]] == group]

            # Normalize the values for bubble size
            values_scaled = scaler.fit_transform(df_group[df_group.columns[2]].values.reshape(-1,1)) * 1000

            # Bubble plot for each group
            ax = fig.add_subplot(gs[i, 0])
            scatter = ax.scatter(x=df_group[df_group.columns[1]], y=df_group[df_group.columns[2]],
                                 s=values_scaled,
                                 c=df_group[df_group.columns[2]], cmap='viridis', alpha=0.5)
            ax.set_title(f'Bubble Plot for Group {group}')
            ax.set_xlabel(df_group.columns[1])
            ax.set_ylabel(df_group.columns[2])

            # Add colorbar
            plt.colorbar(scatter, ax=ax)

        # Set the title of the figure
        fig.suptitle(self.title)

        plt.tight_layout()
        plt.show()

# Assuming you have a dataframe df
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6000, 7000, 8000, 9000, 6000, 7000, 8000, 9000]
})

obj = MyClass(title='My Bubble Plots')
obj.draw_plots(df)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MyClass:
    def __init__(self, title):
        self.title = title

    def draw_plots(self, df):
        fig = plt.figure(figsize=(10, 10))

        # Create a colormap for the unique groups
        colormap = plt.cm.get_cmap('viridis', len(df[df.columns[0]].unique()))

        # Normalize the values for bubble size
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(df[df.columns[2]].values.reshape(-1,1)) * 1000

        # Bubble plot for all data
        scatter = plt.scatter(x=df[df.columns[1]], y=df[df.columns[2]],
                             s=values_scaled,
                             c=df[df.columns[0]].astype('category').cat.codes, cmap=colormap, alpha=0.5)

        plt.title(self.title)
        plt.xlabel(df.columns[1])
        plt.ylabel(df.columns[2])

        # Create a legend for the colors
        colorbar = plt.colorbar(scatter, ticks=range(len(df[df.columns[0]].unique())))
        colorbar.set_label(df.columns[0])
        colorbar.set_ticks([i + 0.5 for i in range(len(df[df.columns[0]].unique()))])
        colorbar.set_ticklabels(df[df.columns[0]].unique())

        plt.show()

# Assuming you have a dataframe df
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6000, 7000, 8000, 9000, 6000, 7000, 8000, 9000]
})

obj = MyClass(title='My Bubble Plot')
obj.draw_plots(df)

import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MyClass:
    def __init__(self, title):
        self.title = title

    def draw_plots(self, df):
        # Normalize the values for bubble size
        scaler = MinMaxScaler()
        df[df.columns[2]] = scaler.fit_transform(df[df.columns[2]].values.reshape(-1,1)) * 1000

        # Bubble plot for all data
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=df, x=df.columns[1], y=df.columns[2], hue=df.columns[0], size=df.columns[2], legend="full", sizes=(50, 500))
        plt.title(self.title)

        plt.show()

# Assuming you have a dataframe df
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6000, 7000, 8000, 9000, 6000, 7000, 8000, 9000]
})

obj = MyClass(title='My Bubble Plot')
obj.draw_plots(df)


import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MyClass:
    def __init__(self, title):
        self.title = title

    def draw_plots(self, df):
        # Normalize the values for bubble size
        scaler = MinMaxScaler()
        df[df.columns[2]] = scaler.fit_transform(df[df.columns[2]].values.reshape(-1,1)) * 1000

        # Bubble plot for all data
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=df, x=df.columns[1], y=df.columns[2], hue=df.columns[0], size=df.columns[2], palette="Pastel1", legend="full", sizes=(50, 500))
        plt.title(self.title)

        plt.show()

# Assuming you have a dataframe df
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6000, 7000, 8000, 9000, 6000, 7000, 8000, 9000]
})

obj = MyClass(title='My Bubble Plot')
obj.draw_plots(df)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# List of metrics and time periods
metrics = ['A', 'B', 'C', 'D', 'E', 'F']
time_periods = ['t2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2022Q2', 't2022Q3', 't2022Q4']

# Create a DataFrame with random data
np.random.seed(0)  # for reproducible results
data = np.random.rand(len(metrics), len(time_periods))
df = pd.DataFrame(data, index=metrics, columns=time_periods)

# Create a heatmap
sns.heatmap(df, cmap='viridis')
plt.title('Heatmap of Metrics over Time Periods')
plt.show()

from sklearn.preprocessing import MinMaxScaler
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
class MyClass:
    def __init__(self, title):
        self.title = title
    def draw_plots(self, df):
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 3, figure=fig)

        # Create a colormap for the unique groups
        colormap = plt.cm.get_cmap('viridis', len(df[df.columns[0]].unique()))

        # Normalize the values for bubble size
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(df[df.columns[2]].values.reshape(-1,1)) * 1000

        # Bubble plot
        ax_bubble = fig.add_subplot(gs[0, 2])
        scatter = ax_bubble.scatter(x=df[df.columns[1]], y=df[df.columns[2]],
                        s=values_scaled,
                        c=df[df.columns[0]].astype('category').cat.codes, cmap=colormap, alpha=0.5)

        # Create a legend for the colors
        colorbar = plt.colorbar(scatter, ticks=range(len(df[df.columns[0]].unique())), ax=ax_bubble)
        colorbar.set_label(df.columns[0])
        colorbar.set_ticks([i + 0.5 for i in range(len(df[df.columns[0]].unique()))])
        colorbar.set_ticklabels(df[df.columns[0]].unique())

        # Line plot
        ax_line = fig.add_subplot(gs[1, :])
        sns.lineplot(data=df, x=df.columns[1], y=df[df.columns[2]], hue=df.columns[0], ax=ax_line)

        # Stacked bar plot
        df_pivot = df.pivot(index=df.columns[1], columns=df.columns[0], values=df.columns[2])
        ax_bar = fig.add_subplot(gs[0, 0])
        df_pivot.plot(kind='bar', stacked=True, ax=ax_bar)

        # Define a custom colormap
        c_map = LinearSegmentedColormap.from_list('mycmap', ['yellow', 'green'])

        # Heatmap
        ax_heat = fig.add_subplot(gs[0, 1])
        sns.heatmap(df_pivot, cmap=c_map, ax=ax_heat)

        # Set the title of the figure
        fig.suptitle(self.title)

        plt.tight_layout()
        plt.show()
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time_period': [1, 2, 1, 2, 3, 4, 3, 4],
    'value': [6000, 7000, 8000, 9000, 6000, 7000, 8000, 9000]
})

obj = MyClass(title='My Bubble Plot')
obj.draw_plots(df)