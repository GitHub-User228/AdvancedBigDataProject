import matplotlib.pyplot as plt
import seaborn as sns

def simple_plot(x, y, xlabel, ylabel, scatterplot=False, lineplot=True,
                params_dict={'figure.figsize': (20,8), 'font.size': 20, 'lines.linewidth': 5, 'lines.markersize': 15}):
    if params_dict is not None:
        plt.rcParams.update(params_dict)
    plt.figure()
    sns.set_style("darkgrid")
    if lineplot: sns.lineplot(x=x, y=y)
    if scatterplot: sns.scatterplot(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()    