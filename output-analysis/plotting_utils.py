import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_multi_hist(cols, labels, num_bins = 100, title=None, xlims=None, ylims=None, show=False, save_to=None):
    if xlims is None:
        min_x = min([min(col) for col in cols])
        max_x = max([max(col) for col in cols])
        bins = np.linspace(min_x, max_x, num_bins)
    else:
        bins = np.linspace(*xlims, num_bins)
    for col, label, in zip(cols, labels):
        plt.hist(col, bins, label=label, alpha=1/2)
    plt.legend(loc='upper right')
    if xlims is not None:
        plt.xlim(*xlims)
    if ylims is not None:
        plt.ylim(*ylims)
    if title is not None:
        plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()