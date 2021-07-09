import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt

#==========#==========#==========#==========#==========#==========#==========#


def t_SNE(X,
          y=None,
          init='pca',
          n_iter=1000,
          verbose=1,
          random_state=501,
          is_norm=True,
          is_pos=True):
    from sklearn import manifold
    # t-SNE
    tsne = manifold.TSNE(n_components=2,
                         init=init,
                         n_iter=n_iter,
                         verbose=verbose,
                         random_state=random_state)
    X_tsne = tsne.fit_transform(X, y)

    def fn_norm(X):
        x_min = np.min(X, 0)
        x_max = np.max(X, 0)
        return (X - x_min) / (x_max - x_min)

    def fn_pos(X):
        return {i: (float(c), float(r)) for i, (r, c) in zip(range(len(X)), X)}

    if is_norm:
        X_tsne = fn_norm(X_tsne)

    if is_pos:
        X_tsne = fn_pos(X_tsne)

    return X_tsne
#==========#==========#==========#==========#==========#==========#==========#


def get_hsv_to_str(h=1, s=1, v=1):
    import colorsys
    def fn(x): return hex(int(x * 255) // 16)[2:] + hex(int(x * 255) % 16)[2:]
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    str_color = '#' + fn(r) + fn(g) + fn(b)
    return str_color


def get_hsv_str_colors(num_color: int, s=1, v=1):
    str_colors = []
    for i in range(num_color):
        str_color = get_hsv_to_str(i / num_color, s, v)
        str_colors.append(str_color)
    return str_colors


def get_num_str_colors(num_color: int, is_default=True):
    # case 1: using default colors
    default_colors = [
        '#ff0000', '#FFD700', '#48ff00', '#00FFFF', '#9400D3', '#D2691E', '#ff00da'
    ]
    num_default = len(default_colors)
    if is_default and num_color <= num_default:
        str_colors = default_colors[:num_color]
        return np.array(str_colors)
    # case 2: using hsv colors
    str_colors = get_hsv_str_colors(num_color)
    return np.array(str_colors)

#==========#==========#==========#==========#==========#==========#==========#
