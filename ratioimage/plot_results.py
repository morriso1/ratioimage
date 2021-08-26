# AUTOGENERATED! DO NOT EDIT! File to edit: 01_plot_results.ipynb (unless otherwise specified).

__all__ = ['tidy_create_strip_box_plot', 'tidy_create_swarm_box_plot']

# Cell
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Cell
def tidy_create_strip_box_plot(
    y_axis_start: int = 0,
    y_axis_limit: int = None,
    y_label: str = "set y label",
    notch: bool = True,
    **kwargs,
):
    """Creates sns plots. Pass **kwargs to sns.stripplot and sns.boxplot."""

    ax = sns.stripplot(
        alpha=0.2, zorder=0, jitter=0.3, edgecolor="gray", linewidth=0.5, **kwargs
    )
    ax = sns.boxplot(
        fliersize=0, zorder=1, saturation=0.9, linewidth=1.5, notch=notch, **kwargs
    )

    if y_axis_limit is not None:
        ax.set_ylim(top=y_axis_limit)

    ax.set_ylim(bottom=y_axis_start)
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    ax.xaxis.set_label_text("")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.setp(ax.spines.values(), linewidth=1)
    sns.despine()

# Cell
def tidy_create_swarm_box_plot(
    y_axis_start: int = 0,
    y_axis_limit: int = None,
    y_label: str = "set y label",
    **kwargs,
):
    """Creates sns plots. Pass **kwargs to sns.swarmplot and sns.boxplot."""

    ax = sns.swarmplot(
        alpha=0.8, zorder=1, edgecolor="gray", linewidth=0.5, size=5, **kwargs
    )

    ax = sns.boxplot(
        fliersize=0, zorder=0, saturation=0.9, linewidth=1.5, notch=False, **kwargs
    )

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))

    if y_axis_limit != None:
        ax.set_ylim(top=y_axis_limit)

    ax.set_ylim(bottom=y_axis_start)
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    ax.xaxis.set_label_text("")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.setp(ax.spines.values(), linewidth=1)
    sns.despine()