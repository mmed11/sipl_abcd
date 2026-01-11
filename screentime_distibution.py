import reader
import extras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product
from typing import Tuple, Iterable
from pandas import DataFrame as df
from matplotlib.pyplot import get_cmap
from directories import figuresDirectory



def plot_dataframe_bars(
    df: df,
    n_bins: int | None = None,
    colormap: str = 'tab10',
    title: str | None = None,
    figsize: Tuple[int, int] | None = None,
    n_rows: int = 1,
    n_cols: int | None = None,
    dpi: int = 100,
    show: bool = False,
    path: Path | str | None = None,
    x_log: bool = False,
    y_log: bool = False,
    x_label: str | None = None
):
    """
    Plot a bar plot subplot for each column in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing numeric data to plot.
    bin_width : float, default 1.0
        The interval width between bins for the bar plots.
    colormap : str, default 'tab10'
        Name of the matplotlib colormap to color each subplot differently.
    figsize : tuple, optional
        Figure size, e.g., (10, 6). If None, it is automatically determined.
    title : str, optional
        Overall title for the figure.
    """
    
    n_plots = df.shape[1]

    if x_label is None:
        x_label = 'Value' + ' (Logarithmic)' if x_log else ''
    y_label = 'Count' + ' (Logarithmic)' if y_log else ''

    if n_cols is None:
        n_cols = n_plots 

    if figsize is None:
        figsize = (n_cols * 7, n_rows * 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    cmap = get_cmap(colormap)
    
    for i, col in enumerate(df.columns):

        ax = axes[i]
        ax.set_title(col)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        data = df[col]
        bins = None

        if n_bins is not None:
            if x_log:
                bins = np.logspace(np.log10(data[data > 0].min()), np.log10(df[col].max()), n_bins)
            else:
                bins = np.linspace(0, df[col].max(), n_bins)
        
        _, bins, _ = ax.hist(df[col], bins=bins, color=cmap(i % cmap.N), edgecolor='black')
        ax.set_xticks(bins[::2])

        try:
            if x_log:
                ax.set_xscale('log')
            if y_log:
                ax.set_yscale('log')
        except Exception:
            ax.clear()
            ax.set_xscale('linear')
            ax.set_yscale('linear')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=dpi)

    if show:    
        plt.show()


if __name__ == '__main__':

    screentime = reader.readScreentimeData()
    screentime = pd.concat([screentime] + [extras.generalize_logs_data(screentime)], axis='columns')
    metrics = reader.readGraphMetrics()

    screentime = screentime[screentime['participant_id'].isin(metrics['participant_id'])].reset_index(drop=True)
    screentime = screentime.loc[:, screentime.columns.difference(['participant_id', 'session_id'])]

    n_bins = 24
    n_rows = 8
    n_cols = 7
    dpi = 200
    colormap = 'tab10'
    x_label = 'Avg. usage (min/day)'

    col_avgs = screentime.mean()
    sorted_cols = col_avgs.sort_values(ascending=False).index
    screentime_sorted = screentime[sorted_cols]

    datasets = [screentime, screentime_sorted]
    x_logs = [False, True]
    y_logs = [False, True]

    for dataset, x_log, y_log in product(datasets, x_logs, y_logs):

        filename = 'screentime_distribution_'
        filename += 'alphabetical' if dataset is screentime else 'sorted'
        filename += '_xlog' if x_log else ''
        filename += '_ylog' if y_log else ''
        filename += '.png'

        plot_dataframe_bars(
            dataset, n_bins, n_rows=n_rows, n_cols=n_cols, x_log=x_log, y_log=y_log,
            dpi=dpi, x_label=x_label, path=figuresDirectory.joinpath(filename)
        )
        

    '''
    plot_dataframe_bars(
        screentime, n_bins, n_rows=n_rows, n_cols=n_cols, dpi=dpi, x_label=x_label,
        path=figuresDirectory.joinpath('screentime_distibution_alphabetical.png')
    )    
    plot_dataframe_bars(
        screentime, n_bins, n_rows=n_rows, n_cols=n_cols, dpi=dpi, y_log=True, x_label=x_label,
        path=figuresDirectory.joinpath('screentime_distibution_alphabetical_ylog.png')
    )
    plot_dataframe_bars(
        screentime, n_bins, n_rows=n_rows, n_cols=n_cols, dpi=dpi, y_log=True, x_log=True, x_label=x_label,
        path=figuresDirectory.joinpath('screentime_distibution_alphabetical_ylog_xlog.png')
    ) 
    
   

    plot_dataframe_bars(
        screentime, n_bins, n_rows=n_rows, n_cols=n_cols, dpi=dpi, x_label=x_label,
        path=figuresDirectory.joinpath('screentime_distibution_sorted.png')
    )    
    plot_dataframe_bars(
        screentime, n_bins, n_rows=n_rows, n_cols=n_cols, dpi=dpi, y_log=True, x_label=x_label,
        path=figuresDirectory.joinpath('screentime_distibution_sorted_ylog.png')
    ) 
    plot_dataframe_bars(
        screentime, n_bins, n_rows=n_rows, n_cols=n_cols, dpi=dpi, y_log=True, x_log=True, x_label=x_label,
        path=figuresDirectory.joinpath('screentime_distibution_sorted_ylog_xlog.png')
    ) 
    '''