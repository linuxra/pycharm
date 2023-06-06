import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self, df, id_vars):
        self.df = df.melt(id_vars=id_vars, var_name='time', value_name='value')
        self.grouped = self.df.groupby(id_vars)

    def plot_data(self):
        num_descriptions = len(self.grouped)
        fig = plt.figure(figsize=(15, 5 * num_descriptions))

        for idx, (desc, group) in enumerate(self.grouped):
            ax_plot = plt.subplot2grid((num_descriptions + 2, 3), (1 + idx // 3, idx % 3))
            ax_plot.plot(group['time'], group['value'])
            ax_plot.set_title(f'Plot of {desc}')
            ax_plot.set_xlabel('Time')
            ax_plot.set_ylabel('Value')

        plt.tight_layout()
        plt.show()

np.random.seed(42)
data = np.random.randint(low=int(1e6), high=int(1e9), size=(10, 36))
ids = [f'id{i}' for i in range(1, 11)]
desc = ['desc'+str(i) for i in range(1, 11)]
times = [f't{i}' for i in range(1, 37)]
df = pd.DataFrame(data, columns=times)
df.insert(0, 'id', ids)
df.insert(1, 'description', desc)

id_vars = ['id', 'description']
plotter = DataPlotter(df, id_vars)
plotter.plot_data()

