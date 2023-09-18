from unittest.mock import patch

import pytest
from tga.file_handling import TGAExperiment, TGAParser


def test_valid_content():
    # Mock the content of a valid TGA file
    content = (
                "# Export date and time: Mi Sep 6 09:43:56 2023\n"
                "# Name: V1\n"
                "# Measurement date and time: So Aug 20 21:08:48 2023\n"
                "# Weight: 1032 mg\n")

    with patch('tga.file_handling.TGAParser._read_file_header', return_value=content):
        experiment = TGAParser.from_file("test_data/1041_V1.txt")
        assert isinstance(experiment, TGAExperiment)
        assert experiment.name == "V1"

        from datetime import datetime

        expected_date = datetime(2023, 8, 20, 21, 8, 48)
        assert experiment.measurement_date == expected_date
        assert experiment.weight == 1032


def test_invalid_content():
    # Mock the content of an invalid TGA file
    content = (
                "# Export date and time: Mi Sep 6 09:43:56 2023\n"
                "# Name: V1\n"
                "# Measurement date and time: So Aug 20 21:8:48 2023\n"
                "# Weight: 1030 mg\n")

    with patch('tga.file_handling.TGAParser._read_file_header', return_value=content):
        with pytest.raises(ValueError):
            TGAParser.from_file("test_data/1041_V1.txt")


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_Texperiment = TGAParser.from_file("non_existent_path.txt")

def test_partial_content():
    # Mock the content of a TGA file with missing lines
    content = (
        "# Name: V1\n"
        "# Measurement date and time: So Aug 20 21:08:48 2023\n")

    with patch('tga.file_handling.TGAParser._read_file_header', return_value=content):
        with pytest.raises(ValueError):
            experiment = TGAParser.from_file("test_data/1041_V1.txt")
