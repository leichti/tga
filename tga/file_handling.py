import re
from datetime import datetime
from itertools import islice
from pathlib import Path

import pandas as pd


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
        r"# Weight: (?P<weight>\d+) mg$"
    )

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
        with path.open() as file:
            return ''.join(line for line in file.readlines()[:4])

    @classmethod
    def _parse(cls, header, data):
        match = cls.PATTERN.search(header)
        if not match:
            raise ValueError("Expected TGA format not matched.")

        name = match.group("name")
        measurement_date = match.group("measurement_date")
        weight = int(match.group("weight"))

        if len(data) == 0:
            raise ValueError("No experimental data found.")

        return TGA_Experiment(name, measurement_date, weight, data)

    @classmethod
    def _read_file_content(cls, path):
        return pd.read_csv(path, delimiter=",", skiprows=4, encoding="ISO-8859-1")


class TGA_Experiment:

    GERMAN_TO_ENGLISH_DAYS = {
        'Mo': 'Mon',
        'Di': 'Tue',
        'Mi': 'Wed',
        'Do': 'Thu',
        'Fr': 'Fri',
        'Sa': 'Sat',
        'So': 'Sun'
    }

    def __init__(self, name, measurement_date, weight, data):
        self.name = name
        self.measurement_date = self._parse_date(measurement_date)
        self.weight = float(weight)
        self.data = data

    def __str__(self):
        return (f"Name: {self.name}\n"
                f"Measurement Date/Time: {self.measurement_date}\n"
                f"Weight: {self.weight} mg")

    def _parse_date(self, date_str):
        for german, english in self.GERMAN_TO_ENGLISH_DAYS.items():
            date_str = date_str.replace(german, english)
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")


# For testing
if __name__ == "__main__":
    path = "../examples/2023_waelz_reduction/data/1041_V1.txt"
    try:
        tga_experiment = TGAParser.from_file(path)
        print(tga_experiment.data)
    except Exception as e:
        print(e)

