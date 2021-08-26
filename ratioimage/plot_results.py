# AUTOGENERATED! DO NOT EDIT! File to edit: 01_plot_results.ipynb (unless otherwise specified).

__all__ = ['tidy_create_strip_box_plot']

# Cell
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Cell
def tidy_create_strip_box_plot(
    x_figSize: int = 2.5,
    y_figSize: int = 2.5,
    scale_size: int = 1,
    y_axis_start: int = 0,
    y_axis_limit: int = None,
    y_label: str = "set y label",
    save_path: str = None,
    notch: bool = True,
    **kwargs,
):
    """Creates sns plots. Pass **kwargs to sns.stripplot and sns.boxplot."""

    fig, ax = plt.subplots()
    fig.set_size_inches(x_figSize * scale_size, y_figSize * scale_size)

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

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")

    return fig