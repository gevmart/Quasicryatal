import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def annotate(fig, ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def heatmap(data, x, y, ax=None, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('viridis')(range(ncolors))

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1, ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='alpha', colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

    extent = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
    im = ax.imshow(data, extent=extent, **kwargs)

    if cbarlabel != "":
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    return im
