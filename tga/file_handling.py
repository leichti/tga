import re
import time
import traceback
from datetime import datetime
from itertools import islice
from pathlib import Path
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def read_files_in_directory(path):
    """
    Reads the given directory and returns a list of dictionaries with file names and their formats.

    :param path: Path to the directory to be read.
    :return: List of dictionaries with "name" and "format" keys.
    """
    directory = Path(path)
    files = [f for f in directory.iterdir() if f.is_file()]

    result = []
    for file in files:
        result.append({
            "name": file.stem,
            "extension": file.suffix[1:]  # remove the leading dot from the extension
        })

    return result

def get_files_by_extension(path, extension):
    """
    Returns a list of files with the given format from the specified directory.

    :param path: Path to the directory.
    :param extension: Desired file format (e.g., "txt").
    :return: List of file names with the specified format.
    """
    files = read_files_in_directory(path)
    return [".".join([file["name"], file["extension"]]) for file in files if file["extension"] == extension]


class TGAParser:

    PATTERN = re.compile(
        r"^# Export date and time: [A-Za-z]{2} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4}\n"
        r"# Name: (?P<name>[A-Za-z0-9]+)\n"
        r"# Measurement date and time: (?P<measurement_date>[A-Za-z]{2} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4})\n"
        r"# Weight: (?P<weight>\d+(\.\d+)?) mg$"
    )

    LOOKUP = {"Temperature(°C)": {"new": "T", "dtype":"float32"}, "TEMP_NOM_SAMPLE(K)":{"new":"T_target", "dtype":"float32"}, "Delta m(mg)": {"new": "dm", "dtype":"float32"},
              "Time(s)": {"new": "t", "dtype":"float32"}, "Wasser(ml/min)": {"new": "h2o", "dtype":"float32"},
              "Gas 1(sccm/min)": {"new": "gas1", "dtype":"float32"}, "Gas 2(sccm/min)": {"new": "gas2", "dtype":"float32"},
              "Purge(sccm/min)": {"new": "purge", "dtype":"float32"}, "TEMP_FURNACE(K)": {"new": "T_furnace", "dtype":"float32"}}

    MIN_HEIGHT_PCT_DEFAULT = 0.1


    @classmethod
    def from_file(cls, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: '{path.resolve()}'")

        header = cls._read_file_header(path)
        content = cls._read_file_content(path)
        return cls._parse(header, content)

    @staticmethod
    def _read_file_header(path):
        with path.open(encoding='cp1252') as file:
            return ''.join(line for line in file.readlines()[:4])

    @classmethod
    def _parse(cls, header, data):
        match = cls.PATTERN.search(header)
        if not match:
            raise ValueError("Expected TGA format not matched.")

        name = match.group("name")
        measurement_date = match.group("measurement_date")
        weight = float(match.group("weight"))

        if len(data) == 0:
            raise ValueError("No experimental data found.")

        return TGAExperiment(name, measurement_date, weight, data)

    @classmethod
    def _read_file_content(cls, path):

        keys = list(TGAParser.LOOKUP.keys())
        dtypes = {key: TGAParser.LOOKUP[key]['dtype'] for key in keys}
        names = {key: TGAParser.LOOKUP[key]['new'] for key in keys}

        df =  pd.read_csv(path, delimiter=",", skiprows=4, encoding="ISO-8859-1", usecols=TGAParser.LOOKUP.keys(), dtype=dtypes)
        df.rename(columns=names, inplace=True)

        for T_column in ["T_furnace", "T_target"]:
            df[T_column] -= 273.15

        return df

class TGAExperiment:

    GERMAN_TO_ENGLISH_DAYS = {
        'Mo': 'Mon',
        'Di': 'Tue',
        'Mi': 'Wed',
        'Do': 'Thu',
        'Fr': 'Fri',
        'Sa': 'Sat',
        'So': 'Sun'
    }

    SAVGOL_WINDOW_SIZE = 131

    def __init__(self, name, measurement_date, weight, data):
        self.name = name
        self.measurement_date = self._parse_date(measurement_date)
        self.weight = float(weight)
        self.data = data
        self.raw_segments = []

    def __str__(self):
        return (f"Name: {self.name}\n"
                f"Measurement Date/Time: {self.measurement_date}\n"
                f"Weight: {self.weight} mg")

    def _parse_date(self, date_str):
        for german, english in self.GERMAN_TO_ENGLISH_DAYS.items():
            date_str = date_str.replace(german, english)
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")

    def search_segments_by_gases(self, gases, *args, **kwargs):
        for gas in gases:
            self.search_segments_by_gas(gas, *args, **kwargs)

    def search_segments_by_gas(self, column, min_height_pct=0.1):
        dydt = self._calculate_derivative(column)
        min_height = self._determine_min_height(dydt, min_height_pct)
        peaks = self._find_peaks(dydt, min_height)
        self._process_peaks(peaks)

    def _calculate_derivative(self, column):
        return savgol_filter(self[column], self.SAVGOL_WINDOW_SIZE, 1, deriv=1)

    def _determine_min_height(self, dydt, min_height_pct):
        min_height = max(dydt) * min_height_pct
        return max(min_height, 0.1)

    def _find_peaks(self, dydt, min_height):
        on_peaks = scipy.signal.find_peaks(dydt, height=min_height)[0]
        off_peaks = scipy.signal.find_peaks(-dydt, height=min_height)[0]
        peaks = np.concatenate([on_peaks, off_peaks])
        return np.sort(peaks)

    def _process_peaks(self, peaks):
        start_indexes, stop_indexes = self._get_start_stop_indexes(peaks)

        for i, (on_idx, off_idx) in enumerate(zip(start_indexes, stop_indexes)):
            segment_data = self.data.iloc[on_idx:off_idx].copy()

            if segment_data.empty:
                raise ValueError(f"Cannot add empty segment. Index range: {on_idx}:{off_idx}")

            self.raw_segments.append(Segment(segment_data, self))

    def _get_start_stop_indexes(self, peaks):
        start_indexes = [peaks[i] for i in range(len(peaks)) if i % 2 == 0]
        stop_indexes = [peaks[i] for i in range(len(peaks)) if i % 2 != 0]

        if len(start_indexes) > len(stop_indexes):
            stop_indexes.append(len(self["t"]) - 1)

        if len(stop_indexes) > len(start_indexes):
            start_indexes.insert(0, 0)

        return start_indexes, stop_indexes

    def __getitem__(self, item):
        return self.data[item]

class Segment:

    def __init__(self, df, experiment):
        self.data = df
        self.experiment = experiment
        self.t0 = self.data["t"].iloc[0]
        self.description = []

    def add_description(self, description):
        if description in self.description:
            return None

        self.description.append(description)



# For testing
if __name__ == "__main__":
    path = "../tests/test_data/1073_V18.txt"
    try:
        start = time.time_ns()
        tga_experiment = TGAParser.from_file(path)

        # @todo implement this feature
        tga_experiment.search_segments_by_gases(["gas1", "gas2", "purge"])
        #tga_experiment.search_gas_segments(["gas_col1", "gas_col2", "gas_col3"])
        #tga_experiment.search_temperature_segments(["T"])

        segments_list = tga_experiment.raw_segments

        segments_list = sorted(segments_list, key=lambda segment: segment.t0)

        pass

    except Exception as e:
        traceback.print_exc()
