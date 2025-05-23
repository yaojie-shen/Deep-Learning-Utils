# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 13:56
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : plot.py

__all__ = ["plot_distribution"]

import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(data, remove_outlier=False, percent_range=(0.1, 99.9)):
    # remove outlier
    if remove_outlier:
        data = np.array(data)
        lower, higher = np.percentile(data, percent_range, axis=0)
        data = data[(lower < data) & (data < higher)]

    fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True, dpi=200)
    ax.violinplot(data, vert=False, showmeans=False, showmedians=False, showextrema=False, points=200)
    ax.set_ylim(1)
    fig.show()
