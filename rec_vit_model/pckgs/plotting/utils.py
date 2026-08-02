import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap


def get_colormap(n_items, scheme='jet', grey_intensity=0.5):
    """
    Allows to create gradient-like colormaps
    For further improvement and usage, see https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    if scheme == 'grey':
        newmap = [(grey_intensity, grey_intensity, grey_intensity, 1) for _ in range(n_items)]
    else:
        newmap = []
        cmap = colormaps[scheme]
        for c in np.linspace(0, 1, num=n_items):
            newmap.append(cmap(c))
    return ListedColormap(newmap)
