import matplotlib.pyplot as plt
from math import sqrt, ceil
from typing import List, Tuple, Optional
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from PIL import Image
from numpy import ndarray
from natsort import natsorted
import os

class Plot2D:
    def plot(self, xs: List[List[float]], ys: List[List[float]], savepath: str, **kwargs) -> None:
        parent = savepath[0:savepath.rfind('/')]
        if not os.path.exists(parent):
            os.makedirs(parent)
            
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
        markersize = kwargs['markersize'] if 'markersize' in kwargs else 2
        linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 1

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
            axes[i].plot(x, y, marker=marker, linestyle=linestyle, markersize=markersize, linewidth=linewidth)
            axes[i].set_title(f"{title}: {i}")
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].axhline(y[-1], linestyle='--', color='r', linewidth=linewidth)
            axes[i].text(-5, y[-1], '%.2f'%y[-1], va='center', ha='right', color='r')
        
        plt.savefig(savepath, dpi=dpi)


class TensorToImage:
    def __init__(self, image_size: Tuple[int], mode: Optional[str]="RGB") -> None:
        self._image_size_ = image_size
        self._mode_ = mode
        self.__toPILImage__ = ToPILImage(mode=self._mode_)

    def convert(self, tensor: Tensor, n_samples: int = 30, n_rows: int = 6) -> Image.Image:
        image_tensor = tensor if tensor.device.type == 'cpu' else tensor.detach().cpu()
        image_tensor = image_tensor.view(-1, *self._image_size_)[:n_samples]
        image_grid = make_grid(image_tensor, nrow=n_rows).squeeze()
        return self.__toPILImage__(image_grid)
    
    def toGIF(self, path: str, remove_cache: bool=True, gif_name: str="merged", **kwargs):
        for root, dirs, files in os.walk(path):
            pass
        gif = [Image.open(path + file) for file in natsorted(files)]
        gif[0].save(path + f"{gif_name}.gif", save_all=True, append_images=gif[1:], **kwargs)

        if remove_cache:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.png'):
                        os.remove(os.path.join(root + file))