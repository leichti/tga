import os
import sys
from collections.abc import Iterable

import pandas as pd
import re
import numpy as np
import pywt
import scipy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
import datetime

#from h2_model import TimeReactor, GasFlow



#todo drift control. It may be the case that the mass signal drift over time. This could have a (limited) influence on our data


class File:
    
    def __init__(self, filepath, folder=None):
        self.filepath = filepath
        self.folder = folder
        self.reference = None
        self.id = None
        self.filename = filepath.split("/")[-1]
        self.date = None
        self.df = None

    def timestamps(self):
        self.df["t"] += self.date.timestamp()

    def __str__(self):
        return self.filename


class TgaFile(File):

    def __init__(self, *args):
        super(TgaFile, self).__init__(*args)

        self.lookup = {"Temperature(°C)": "T", "Delta m(mg)": "dm", "Time(s)": "t",
                       "Wasser(ml/min)": "h2o", "Gas 1(sccm/min)": "gas1", "Gas 2(sccm/min)": "gas2",
                       "Purge(sccm/min)": "purge", "TEMP_FURNACE(K)": "T_furnace"}

        self.weight = None
        self.h2 = None
        self.repeat = ""

        self.extract()
        self.info()
        self.data()
        self.timestamps()
        #self.load_h2()
        # self.trial = re.match("")

    def extract(self):
        matches = re.findall("^([0-9]{1,})_([A-Z0-9]{1,})(_{0,1}[Stk]{0,}[Fine]{0,}[AM]{0,})(_[0-9]{1,}){0,}(_REF){0,}(\.txt)$", self.filename)

        if not matches:
            raise ValueError(f"File {self.filename} is not a tga file")

        matches = matches[0]

        if not matches:
            raise ValueError(f"File {self.filename} is not a tga file")
        if matches[4] == "":
            self.reference = False
        else:
            self.reference = True

        if matches[3] != "":
            self.repeat = matches[3]

        self.id = matches[1]

    def data(self):
        df = pd.read_csv(self.filepath, delimiter=",", skiprows=4, encoding="ISO-8859-1")
        self.df = df.rename(columns=self.lookup)
        self.df["h2o"] *= 1244
        self.df["T_furnace"] -= 273 # from K to °C
        # df = df[self.lookup.values()]

    def info(self):
        df = pd.read_csv(self.filepath, nrows=3, encoding="ISO-8859-1")
        weight = df.iloc[2, 0]
        self.weight = float(re.findall("[0-9\.]{1,}", weight)[0])
        self.date = TgaDateParser(df.iloc[1, 0])

    def __str__(self):
        if self.reference:
            return f"{self.id} (Reference)"

        return f"{self.id}{self.repeat}"

    def load_h2(self):
        if self.folder is None:
            return False

        self.h2 = self.folder.h2(self)
        if not self.h2:
            self.df["H2"] = np.nan
            return False
        self.df["H2"] = self.h2.match(self.df["t"])


class H2File(File):

    def __init__(self, *args):
        super(H2File, self).__init__(*args)
        self.extract()
        self.data()
        self.h2 = None # H2 concentration from  start
        self.t = None # corresponding time values from start

    def extract(self):
        try:
            y, M, d, h, m, s, id, ref = re.findall(r"([0-9]{4})-([0-9]{2})-([0-9]{2})-([0-9]{2})_([0-9]{2})_([0-9]{2})_([A-Za-z0-9]{1,})[\_AM]{0,3}(_REF){0,1}.csv$", self.filename)[0]
        except Exception:
            print("Not a valid filename")

        date = datetime.datetime(int(y), int(M), int(d), int(h), int(m), int(s))
        self.date = date
        self.id = id

        if ref == "":
            self.reference = False
        else:
            self.reference = True

    def data(self):
        self.df = pd.read_csv(self.filepath, delimiter=";", encoding="ISO-8859-1")
        self.df["t"] += self.date.timestamp()
        return self.df

    def match(self, t):
        f = interp1d(self.df["t"], self.df["H2"], fill_value=0, bounds_error=False)
        return f(t)

    def load(self):

        self.df["H2_smoothed"] = savgol_filter(self.df["H2"], 800, 2)

        dh2 = np.diff(self.df["H2_smoothed"])
        dt = np.diff(self.df["t"])

        dt[dt <= 0] = 0.1
        dh2_dt = np.append(dh2 / dt, 0)

        self.df["dH2dt"] = dh2_dt

        max_val = max(self.df["dH2dt"])
        min_val = min(self.df["dH2dt"])

        start_idx_rough = self.df.index[self.df["dH2dt"] == max_val][0]
        end_idx_rough = self.df.index[self.df["dH2dt"] == min_val][0]
        idx = end_idx_rough
        old_val = 0
        while True:
            val = self.df["H2_smoothed"].loc[idx]

            if val > old_val:
                old_val = val
                idx -= 1
                continue

            break

        end_idx = idx

        idx = start_idx_rough
        while True:

            val = self.df["H2_smoothed"].loc[idx]
            if val > 1:
                idx -= 1
                continue
            break
        start_idx = idx

        self.df["t"] -= self.df["t"].loc[start_idx]
        f_h2 = interp1d(self.df["t"], self.df["H2_smoothed"])

        self.t = np.arange(0, self.df["t"].loc[end_idx]+1)
        self.h2 = f_h2(self.t)*68.3/69.8


def TgaDateParser(date):
    month, d, h, m, s, y = re.findall('([A-Za-z]{3}) ([0-9]{1,2}) ([0-9]{1,2}):([0-9]{1,2}):([0-9]{1,2}) ([0-9]{4})$', date)[0]
    month = ["Jan", "Feb", "Mrz", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"].index(month) + 1
    date = datetime.datetime(int(y), int(month), int(d), int(h), int(m), int(s))

    return date


class Folder():
    suffix = ""

    def __init__(self, path):
        try:
            files = os.listdir(path)
        except FileNotFoundError:
            print("No such directory")

        self.path = path
        self.files = []
        self.load(files)

    def load(self, files):
        for filename in files:
            if not filename.endswith(self.suffix):
                continue

            self.append(self.path+filename)

    def reference_of(self, id):
        for file in self:
            if file.id == id and file.reference is True:
                return file

        return None

    def __iter__(self):
        return iter(self.files)

    def trials(self):
        for file in self:
            if file.reference is False:
                yield file

    def trial_names(self):
        for file in self:
            if file.reference is False:
                yield str(file)

    def select(self, trial_name):
        for file in self.files:
            if str(file) == trial_name:
                return file


class H2Folder(Folder):
    suffix = ".csv"

    def __init__(self, path):
        super(H2Folder, self).__init__(path)


    def append(self, filename):
        self.files.append(H2File(filename, self))


class TgaFolder(Folder):

    suffix = ".txt"

    def __init__(self, path):
        self.h2_folder = None
        super(TgaFolder, self).__init__(path)


    def append(self, filename):
        self.files.append(TgaFile(filename,self))

    def h2(self, trial):
        self.load_h2_folder()
        for h2_file in self.h2_folder.files:
            if h2_file.id == trial.id and h2_file.reference is trial.reference:
                return h2_file

        return False
        print(f"No H2 File for {trial.id}")

    def load_h2_folder(self):
        if not self.h2_folder:
            self.h2_folder = H2Folder(f"{self.path}/h2_sensor/")




class Trial:

    def __init__(self, file, df=None, initialized=False):

        self.initialized = initialized
        self.idx_stop = None
        self.idx_start = None
        self.t_dimension = "min"

        if type(file) is str:
            self.file = TgaFile(file)
        else:
            self.file = file

        if df is None:
            self.df = self.file.df
        else:
            self.df = df

        self.ref = None
        self.reference_applied = False
        self.smoothed_columns = []

        self.weight = self.file.weight
        if not self.initialized:
            self.initialize()

    def __getitem__(self, item):
        return self.df[item]

    def stepsize(self):
        return self.df.loc[1, "t"] - self.df.loc[0, "t"]

    def segment_from_gas(self, column, segment_idx=0, min_height_pct=0.1):
        # function to find on and off points for column, especially for gases

        # derivate it
        dydt = savgol_filter(self.df[column], 101, 1, deriv=1)
        min_height = max(dydt)*min_height_pct
        on_peaks = scipy.signal.find_peaks(dydt, height=min_height)[0]
        off_peaks = scipy.signal.find_peaks(-dydt, height=min_height)[0]

        if len(off_peaks) < len(on_peaks):
            off_peaks = np.append(off_peaks, -1)

        if len(on_peaks) > len(off_peaks):
            on_peaks.insert(0, 0)

        df = None
        for i, (on_idx, off_idx) in enumerate(zip(on_peaks, off_peaks)):
            if i == segment_idx:
                df = self.df.iloc[on_idx:off_idx].copy()

        if df is None:
            raise ValueError(f"There are only {i} segments")

        return Segment(self.file, df, initialized=True)

    def initialize(self):
        self.df["T"] = self.df["T"].astype(float)
        self.df["m"] = self.df["dm"] + self.weight
        if self.t_dimension == "min":
            self.df["t"] = self.df["t"] / 60


        self.initialized = True

    def apply(self, column, fun, new_name=None):
        if new_name is None:
            new_name = f"{column}_{fun}"

        self.df[new_name] = fun(self.df["t"], self.df[column])
        self.smoothed_columns.append(new_name)

    def detect(self):
        raise NotImplementedError("Please implement a method to detect start and endpoint")

    def export(self, trimmed_export=True):
        if trimmed_export:
            return self.df.loc[self.idx_start:self.idx_stop]
        else:
            return self.df

    def get(self, y, x=None, cropped=True):
        """
        Helper function
        """
        if cropped is False:
            start = self.df.index[0]
            stop = self.df.index[-1]
        else:
            start = self.idx_start
            stop = self.idx_stop

        y = self.df.loc[start:stop, y]

        if x is not None:
            x = self.df.loc[start:stop, x]
            return x, y

        return y

    def add_ref(self, reference, apply=True):
        self.ref = reference

        if apply is True:
            self.df["m"] -= reference(self.df["t"])
            self.df["dm"] -= reference(self.df["t"])
            self.reference_applied = True

    def adjust_weight(self):
        # be careful using this because speciman may loss weight during heat-up phase (e.g. moisture)
        correction = self.df["m"].loc[self.idx_start] - self.weight
        self.df["m"] -= correction


class Segment(Trial):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev = None
        self.next = None

class TrialApplyFunction():

    def __str__(self):
        raise NotImplementedError("")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("")

class Deviate(TrialApplyFunction):

    def __init__(self, order=1):
        super().__init__()
        self.order = 1

    def __str__(self):
        return f"deviate_{self.order}"

    def __call__(self, t, y):
        dydt = np.diff(y)/np.diff(t)
        dydt = np.append(dydt, dydt[-1])
        return dydt

class SavgolSmoother(TrialApplyFunction):

    def __init__(self, window_length, polyorder, **kwargs):
        super().__init__()
        self.window_length = window_length
        self.polyorder = polyorder

    def __str__(self):
        return f"savgol_{self.window_length}_{self.polyorder}"

    def __call__(self, t, y):

        return savgol_filter(y, self.window_length, self.polyorder)

class WaveLetDenoising(TrialApplyFunction):
    
    def __init__(self, threshold=50, wavelet="db1", mode="soft", level=4):

        super().__init__()
        self.threshold = threshold
        self.wavelet = wavelet
        self.mode = mode
        self.level = level


    def __str__(self):
        return f"wavelet_{self.threshold}_{self.wavelet}_{self.level}_{self.mode}"

    def __call__(self, t, y):

        # Perform a wavelet decomposition
        coeffs = pywt.wavedec(y, self.wavelet, level=self.level)

        # Set a threshold value to zero out coefficients that are considered noise

        coeffs_thresholded = [pywt.threshold(i, self.threshold, mode=self.mode) for i in coeffs]
        y_smoothed = (pywt.waverec(coeffs_thresholded, self.wavelet) )
        y_smoothed += y[0]- y_smoothed[0]

        while len(y_smoothed) < len(y):
            y_smoothed.append(y_smoothed[-1])

        while len(y_smoothed) > len(y):
            y_smoothed = y_smoothed[:-1]

        # Reconstruct the signal
        return y_smoothed

class ReductionTrial(Trial):
    def __init__(self, file):
        super().__init__(file)

        self.avg_flow_during_reduction = None # avg overall gasflow during reduction (gas1+gas2+purge+h2o)
        # @todo should do that because otherwise time-values between experiment and reference are not comparable
        #  (e.g. different T start)

        self.detect()

        # let's say that dm = 0 at the start of a trial
        self.df["dm"] -= self.df.loc[self.idx_start, "dm"]

        self.deviate()

    def deviate(self):
        """
        Deviates mass signal with respect to the time
        """
        self.df["dmdt"] = savgol_filter(np.diff(self.df["dm"], prepend=0) / np.diff(self.df["t"], prepend=-1), 200, 3)
        self.df["dm_remaining"] = -(self.df["dm"] - min(self.df["dm"]))

    def detect(self):
        """
        Detects start of a reduction trial by the ramps of the reduction gas flow. Therefore, first deviation of
        the gas flow is calculated, then scipy.signal.find_peaks(gas) is applied to detect positiv ramps (=gas on)
        and scipy.signal.find_peaks(-gas) is applied to detect negative ramps (=gas off)
        """
        self.df["dgasdt"] = np.diff(self.df["gas1"], prepend=0) / np.diff(self.df["t"], prepend=-1)
        self.first_gas_on()
        self.first_gas_off()
        self.shift_to_start()

        self.avg_flow_during_reduction = self.df.loc[self.idx_start:self.idx_stop, ["gas1", "gas2", "purge"]].sum(axis=1).mean()

    def shift_to_start(self):
        self.trial_start = self.df.loc[self.idx_start, "t"]
        self.df["t"] -= self.trial_start

    def first_gas_on(self):
        h2_on_peaks = find_peaks(self.df["dgasdt"], height=self.df["dgasdt"].max() * 0.2, distance=1000)
        self.idx_start = self.df.index[h2_on_peaks[0][0]]

    def first_gas_off(self):

        min_height = (-self.df["dgasdt"]).max()
        max_height = self.df["dgasdt"].max()
        # if no off signal, we find just shit.
        if min_height < self.df["dgasdt"].max() *0.1:
            min_height = (self.df["dgasdt"].max()) *0.1

        h2_off_peaks = find_peaks(-self.df["dgasdt"], height=min_height*0.2, distance=1)
        if len(h2_off_peaks[0]) == 0:
            self.idx_stop = self.df.index[-1]
        else:
            self.idx_stop = self.df.index[h2_off_peaks[0][0] - 100]

        if self.idx_stop < self.idx_start:
            self.idx_stop = self.df.index[-1]


class ReductionReference(ReductionTrial):

    def __init__(self, file):
        super(ReductionReference, self).__init__(file)
        self.df["m"] = self.df["m"] - self.df["m"].iloc[0]

    def __call__(self, t):
        self.f = interp1d(self.df["t"], self.df["m"], bounds_error=False)
        return self.f(t)


class H2():

    def __init__(self, trial, measured):
        # todo would be important to always have corresponding time values
        # todo atm we assume that the int idx of the list equals seconds since "start"
        h2_inflow = GasFlow(int((trial.idx_stop-trial.idx_start)*trial.stepsize()), 1)
        h2_inflow[0:-1] = 1

        time_reactor = TimeReactor(h2_inflow, measured.h2, trial.avg_flow_during_reduction, 0.3)
        time_reactor.calculate()
        self.reactor = time_reactor
        self.effective_h2_flow = h2_inflow.flow
        self.t = range(0, len(self.effective_h2_flow))

        self.f = interp1d(self.t, self.effective_h2_flow, bounds_error=False, fill_value="extrapolate")





