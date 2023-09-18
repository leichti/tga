import re

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import sys
from eval_helper import load_file

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
i = 0

def y_labels(axes):
    axes[6].set_ylabel("Mass")
    axes[5].set_ylabel("Power(%)")
    axes[4].set_ylabel("Gas Flow (ml/(min g)")
    axes[2].set_ylabel("% H2")
    axes[1].set_ylabel("Reaction Rate (mg/(min g)")
    axes[0].set_ylabel("T Furnace")
    axes[7].set_ylabel("Drift [mg]")
    axes[7].set_xlabel("Time [min]")

def over_time(df, axes):
    y_labels(axes)
    x_labels_timed(axes)
    axes[0].plot(df["t"], df["T_furnace"], linestyle="dashed", alpha=0.8, color=colors[i])
    axes[3].plot(df["t"], df["T"], linestyle="dashed", alpha=0.8, color=colors[i])
    axes[1].plot(df["dm_remaining"] / df["m"].iloc[0] * 100, df["dmdt_rel"], linestyle="dashed", alpha=0.8, color=colors[i])
    axes[2].plot(df["t"], df["gas1"] / (df["gas1"] + df["gas2"] + df["purge"]) * 100, linestyle="dashed", alpha=0.8, color=colors[i])
    axes[4].plot(df["t"], (df["gas1"] + df["gas2"] + df["purge"]) / df["m"] * 1000, linestyle="dashed", alpha=0.8, color=colors[i])
    axes[5].plot(df["t"], savgol_filter(df["POWER(%)"], 100, 2), linestyle="dashed", alpha=0.8, color=colors[i])
    axes[6].plot(df["t"], df["m"], linestyle="dashed", alpha=0.8, color=colors[i])
    drift_control(df, axes)

def drift_control(df, axes):
    data = df[df["t"]>100][["t", "m"]].copy()
    if data.empty:
        return False
    data["m"] = data["m"] - data["m"].iloc[0]
    axes[7].plot(data["t"], data["m"], colors[i], alpha=0.8)

def over_progress(df, axes):
    y_labels(axes)
    x_labels_progressed(axes)
    x = df["dm_remaining"]/df["dm"].iloc[-1]+1

    axes[0].plot(x, df["T_furnace"], linestyle="dashed", alpha=0.8, color=colors[i])
    axes[3].plot(x, df["T"], linestyle="dashed", alpha=0.8, color=colors[i])
    axes[1].plot(x, df["dmdt_rel"], linestyle="dashed", alpha=0.8, color=colors[i])
    axes[2].plot(x, df["gas1"] / (df["gas1"] + df["gas2"] + df["purge"]) * 100, linestyle="dashed", alpha=0.8, color=colors[i])
    axes[4].plot(x, (df["gas1"] + df["gas2"] + df["purge"]) / df["m"] * 1000, linestyle="dashed", alpha=0.8, color=colors[i])
    axes[5].plot(x, savgol_filter(df["POWER(%)"], 100, 2), linestyle="dashed", alpha=0.8, color=colors[i])
    axes[6].plot(x, df["m"], linestyle="dashed", alpha=0.8, color=colors[i])
    drift_control(df, axes)



def x_labels_timed(axes):
    for i in [0,2,3,4,5]:
        axes[i].set_xlabel("Time [min]")

    axes[5].set_xlim(0, 5)
    axes[1].set_xlabel("Reaction Progress")

def x_labels_progressed(axes):
    for i in [0,1,2,3,4,5]:
        axes[i].set_xlabel("Reaction Progress")


class Debugger:
    def __init__(self, files = []):
        self.files = files

    def add_file(self, file):
        self.files.append(file)

    def add_files(self, files):
        self.files.extend(files)

    def add_group(self, group):
        self.add_files(group.file_names)

    def plot(self, **kwargs):
        debug_figure(self.files, *kwargs)


def debug_figure(files, x_progress=True, legend_labels=[], name=""):
    global i
    cm = 1 / 2.54
    fig, axes = plt.subplots(3,3, figsize=(33.8*cm,19*cm))
    plt.subplots_adjust(left=0.05, right=0.99, top=0.94)
    axes = axes.flatten()

    if not files:
        return None

    for i, file in enumerate(files):
        if type(file) is str:
            df = load_file(file)
            legend_labels = files
        elif type(file) is pd.DataFrame:
            df = file
        else:
            raise ValueError(f"df is of type {type(df)}. Must be pandas.DataFrame or str")

        ext = "v_time"
        if x_progress:
            ext = "v_progress"
            over_progress(df, axes)
        else:
            over_time(df, axes)

    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    fig.suptitle(name)
    axes[0].legend(legend_labels)
    fig.savefig(f"debug_figures/{name}_{ext}.svg")
    fig.savefig(f"debug_figures/{name}_{ext}.jpg", dpi=600)
    return fig