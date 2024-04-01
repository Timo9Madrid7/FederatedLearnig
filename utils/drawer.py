import matplotlib.pyplot as plt
from math import sqrt, ceil
from typing import List
from numpy import ndarray

class Plot2D:
    def plot(self, xs: List[List[float]], ys: List[List[float]], savepath: str, **kwargs) -> None:        
        num_subplots = len(xs)
        n = ceil(sqrt(num_subplots))
        
        if 'nrows' and 'ncols' in kwargs:
            nrows, ncols = kwargs['nrows'], kwargs['ncols']
        else:
            nrows, ncols = n, n
        figsize = kwargs['figsize'] if 'figsize' in kwargs else [5, 5]
        marker = kwargs['marker'] if 'marker' in kwargs else 'o'
        linestyle = kwargs['linestyle'] if 'linestyle' in kwargs else '-'
        xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else 'x'
        ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else 'y'
        title = kwargs['title'] if 'title' in kwargs else ''
        dpi = kwargs['dpi'] if 'dpi' in kwargs else 300

        subplots = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*n, figsize[1]*n))
        axes: List[plt.Axes] = []

        if isinstance(subplots[1], plt.Axes):
            axes.append(subplots[1])
        elif isinstance(subplots[1], ndarray):
            for axlist in subplots[1]:
                axes.extend(axlist)
        else:
            axes = subplots[1]

        for i, (x, y) in enumerate(zip(xs, ys)):
            axes[i].plot(x, y, marker=marker, linestyle=linestyle)
            axes[i].set_title(f"{title}: {i}")
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)     
        
        plt.savefig(savepath, dpi=dpi)