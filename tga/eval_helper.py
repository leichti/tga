from scipy.interpolate import interp1d
import pandas as pd

import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
#from multiprocessing.dummy import Pool as ThreadPool

# Set the font in the plot
plt.rcParams['font.family'] = "Arial"


from befesa.publication.db import list_by_setting

import numpy as np

def intersection(x, y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


def progress_threader(plot, limit, time_to_95pct):

    def inner(setting_group):
        group = Group(list_by_setting(setting_group), plot, limit_to=limit)
        plot.group(group, add_info=time_to_95pct.new(group, plot))

    return inner


def compare_progress(plot, settings, limit=[0, 11], arrow_altitude=0.66, v_spacing=0.05):
    time_to_95pct = YInfoFactory(95)
    func = progress_threader(plot, limit, time_to_95pct)
    pool = ThreadPool(4)
    pool.map(func, settings)
    pool.close()
    pool.join()

    #for setting_group in settings:
    #    group = Group(list_by_setting(setting_group), plot, limit_to=limit)
    #    plot.group(group, add_info=time_to_95pct.new(group, plot))

    ImprovementFactorWidget(plot, time_to_95pct.list, arrow_altitude=arrow_altitude, v_spacing=v_spacing)
    plot.axes.set_ylim(0,100)
    plot.finish()

def compare_reaction_rate(plot, settings, debug=False):
    for setting_group in settings:
        group = Group(list_by_setting(setting_group), plot.data_selector)
        plot.group(group)
        if debug:
            group.debug()

    plot.axes.set_xlim(0,100)
    plot.finish()

class Group:

    def __init__(self, input, data_selector, loader = None, limit_to=[-np.inf, np.inf], x_name=None, y_names=None):
        """
        Group same experiments for averaging

        :param input: the grouped filenames to be grouped
        :param dataselector: xy values to be selected (e.g. YMasterSelector, AlphaTimeSelector and the like)
        :param loader: the loader to load file names from settings if settings instead of file names are given
        """
        self.x = None
        self.min_x, self.max_x = limit_to
        self.resolution = 200
        self.x_name = None
        self.y_names = None


        self.functions = []
        self.mean_f = None
        self.loader = loader
        self.data_selector = data_selector
        self.files = []

        if type(input) is str:
            self.setting = input
            self.file_names = self.loader(input)
        elif type(input) is list:
            self.setting = "Not defined"
            self.file_names = input

        # need to define one x and one or multiple y values to calc avg values
        self.determine_avg_f()

    def threaded_loading(self, filename):
        file = load_file(filename)
        if self.x_name is None or self.y_names is None:
            x, y = self.data_selector.data_selector(file).xy()
        else:
            x,y = self.data_selector.data_selector(file).get(self.x, self.y)

        self.functions.append(interp1d(x, y, bounds_error=False, fill_value=np.nan))

        if x.min() > self.min_x:
            self.min_x = x.min()
        if x.max() < self.max_x:
            self.max_x = x.max()

    def determine_avg_f(self):
        pool = ThreadPool(4)
        pool.map(self.threaded_loading, self.file_names)
        pool.close()
        pool.join()

        self.x = np.linspace(self.min_x, self.max_x, 200)

        # sum over all trials
        y = np.zeros(len(self.x))
        for f in self.functions:
            y += f(self.x)

        # mean of all trials from sum
        y /= len(self.functions)

        # create an interp1d f for the mean value
        self.mean_f = interp1d(self.x, y, bounds_error=False, fill_value=np.nan)

        return self.mean_f, self.min_x, self.max_x

    def debug(self):
        from debug import debug_figure

        debug_figure(self.file_names, x_progress=True, name=self.setting)


def ImprovementWidget(plot, y_info_list, arrow_altitude, v_spacing, mode="pct"):
    y_info_list.sort(key=lambda x : x.x, reverse=True)
    for i, info in enumerate(y_info_list[:-1]):
        x1 = y_info_list[0].x if mode == "pct" else info.x
        y1 = y_info_list[i].y * (arrow_altitude-i*v_spacing) if mode == "pct" else y_info_list[i].y * (arrow_altitude-i)
        improvement = y_info_list[0].x - y_info_list[i + 1].x if mode == "pct" else y_info_list[0].x - y_info_list[i + 1].x
        x_text = (y_info_list[i].x + y_info_list[i + 1].x) / 2 if mode == "pct" else (x1 + x1 - improvement) / 2

        plot.axes.arrow(x1, y1, -improvement, 0, length_includes_head=True, head_width=1.5, head_length=0.2, color=f"C{i+1}")
        plot.axes.text(x_text, y1 * 1.02, f"-{improvement / y_info_list[0].x * 100:.0f} %", va="bottom", ha="center", color=f"C{i+1}")

def ImprovementFactorWidget(plot, y_info_list, arrow_altitude, v_spacing, mode="pct"):
    y_info_list.sort(key=lambda x : x.x, reverse=False)
    color = len(y_info_list)

    for i, info in enumerate(y_info_list[:-1]):
        x1 = y_info_list[0].x
        y1 = y_info_list[i].y * (arrow_altitude+i*v_spacing)
        increase = y_info_list[0].x - y_info_list[i + 1].x
        increase_factor = y_info_list[i+1].x/y_info_list[0].x
        x_text = (y_info_list[i].x + y_info_list[i + 1].x) / 2

        plot.axes.arrow(x1, y1, -increase, 0, length_includes_head=True, head_width=1.5, head_length=0.2, color=f"C{color-i-2}")
        plot.axes.text(x_text, y1 * 1.02, f"{increase_factor:.1f}x", va="bottom", ha="center", color=f"C{color-i-2}",
                        bbox=dict(boxstyle='square,pad=-0.05', fc='white', alpha=0.8, ec='none'))


class YInfoFactory:

    def __init__(self, y):
        self.y = y
        self.list = []

    def new(self, group, plot):
        new = YInfo(self.y, group, plot)
        self.list.append(new)
        return new

    def sort(self):
        self.list.sort()

class YInfo:

    def __init__(self, y, group, plot):
        self.group = group
        self._plot = plot
        self.y = y
        self.x = 135

    def plot(self):
        x = intersection(self.group.x, self.group.mean_f(self.group.x)-self.y)
        if len(x) == 0:
            return None
        x = x[0]
        self.x = x
        self._plot.axes.plot([self.x, self.x], [0, self.y], linestyle="dashed", linewidth=0.5)

        x_text = self.x
        y_text = self.y/2
        self._plot.axes.text(x_text, y_text, f"{self.x:.1f} min", rotation=90, ha="right", va="center", color=self._plot.color)

class ReductionPlot:

    def __init__(self, title, data_selector, legend_labels=[], ax=None):

        self.legend_lines = []
        self.legend_labels = legend_labels
        self.linestyle = None
        self.alpha = None
        self.color = None
        self.name = title
        self.group_counter = 0

        if not issubclass(data_selector, DataSelector):
            raise AttributeError("Parameter data selector not subclass of DataSelector")

        self.data_selector = data_selector
        if ax is None:
            self.fig, self.axes = plt.subplots(1,1)
        else:
            self.axes = ax
            self.fig = ax.get_figure()

        self.axes.set_title(title)
        self.axes.set_xlim(0, 100)
        self.current_group = None

    def add(self, file=None, x=None, y=None, **kwargs):
        x, y = self.data_selector(file).xy(x, y)
        l, = self.axes.plot(x, y, color=self.color, linestyle=self.linestyle, label="curve", alpha=self.alpha)
        return l

    def xlim(self, *args, **kwargs):
        self.axes.set_xlim(*args, **kwargs)

    def legend(self, *args, **kwargs):
        self.axes.legend(self.legend_lines, self.legend_labels, *args, **kwargs)

    def add_one(self, *args, **kwargs):
        if "linestyle" not in kwargs.keys():
            kwargs["linestyle"] = "dotted"
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 0.7

        self.format_plot(*args, **kwargs)
        self.add(*args, **kwargs)

    def add_mean(self, *args, **kwargs):
        if "linestyle" not in kwargs.keys():
            kwargs["linestyle"] = "solid"
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 1

        self.format_plot(*args, **kwargs)
        l = self.add(*args, **kwargs)
        self.legend_lines.append(l)

    def worker(self, file_name):
        file = load_file(file_name)
        self.add_one(file)
        self.current_group.files.append(file)

    def group(self, group, add_info=None):
        self.current_group = group

        pool = ThreadPool(4)
        pool.map(self.worker, self.current_group.file_names)
        pool.close()
        pool.join()

        x = np.linspace(group.min_x, group.max_x)
        y = group.mean_f(x)
        self.add_mean(x=x, y=y)
        self.xlim(group.min_x, group.max_x)
        self.group_counter += 1

        if add_info:
            add_info.plot()

    def format_plot(self, *args, **kwargs):
        self.alpha = kwargs["alpha"]
        self.linestyle = kwargs["linestyle"]

        if "color" not in kwargs:
            self.color = f"C{self.group_counter}"
        else:
            self.color = kwargs["color"]

    def finish(self):
        self.axes.set_xlabel(self.data_selector.xlabel)
        self.axes.set_ylabel(self.data_selector.ylabel)

        self.legend()

class DataSelector:
    xlabel = ""
    ylabel = ""

    def __init__(self, file):
        self.file = file


    def standard_x_y(self):
        x = self.standard_x()
        y = self.standard_y()
        return x, y

    def standard_x(self):
        # in mg/g
        raise NotImplementedError("Implement Standard X")

    def standard_y(self):
        raise NotImplementedError("Implement Standard Y")

    def xy(self, x=None, y=None):
        if x is None and y is None:
            x, y = self.standard_x_y()
        elif y is None:
            y = self.standard_y()
        elif x is None:
            x = self.standard_x()

        return x, y

class YMasterPlotData(DataSelector):
    xlabel = "Reaction Progress [%]"
    ylabel = "Relative Reaction Rate [%/min]"

    def __init__(self, *args, **kwargs):
        super(YMasterPlotData, self).__init__(*args, **kwargs)

    def standard_x(self):
        # in mg/g
        return self.file["reaction_progress"]

    def standard_y(self):
        return self.file["reaction_progress_rate"]
        #return self.file["dmdt_rel"]

class ProgressOverTime(DataSelector):
    xlabel = "Time [min]"
    ylabel = "Reaction Progress [%]"

    def __init__(self, *args, **kwargs):
        super(ProgressOverTime, self).__init__(*args, **kwargs)

    def standard_x(self):
        # in mg/g
        return self.file["t"]

    def standard_y(self):
        return self.file["reaction_progress"]

def load_file(name, path="data/export/"):
    file = pd.read_csv(f"{path}{name}")

    #max_dmdt = max(-file["dmdt"])
    #t_dmdt_max = float(file[file["dmdt"]==-max_dmdt]["t"])
    #file = file[file["t"] > t_dmdt_max]
    file["dm_remaining"] = -file["dm_remaining"]

    # mass loss in %
    file["dmdt_rel"] = -file["dmdt"] / file["m"] * 1000

    # reaction progress rate [%/min]
    file["reaction_progress_rate"] = file["dmdt"] / file["dm"].iloc[-1] * 100


    dm_rel_remaining = file["dm_remaining"] / file["m"].iloc[0] * 100
    file["reaction_progress"] = 100-dm_rel_remaining / dm_rel_remaining.iloc[0] * 100

    file["flow_rate"] = (file["gas1"] + file["gas2"] + file["purge"] + file["h2o"]) / file["m"] * 1000
    file["h2"] = file["gas1"]/(file["gas1"] + file["gas2"] + file["purge"] + file["h2o"])
    file["h2o"] = file["h2o"]/(file["gas1"] + file["gas2"] + file["purge"] + file["h2o"])
    file["n2"] = (file["gas2"]+file["purge"])/(file["gas1"] + file["gas2"] + file["purge"] + file["h2o"])


    return file
