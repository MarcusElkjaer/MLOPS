import pytest
from pathlib import Path
import pandas as pd
import json
from unittest.mock import patch, MagicMock, mock_open
from reddit_forecast.data import MyDataset, preprocess


def test_my_dataset_initialization():
    """Test initialization of MyDataset with valid data."""
    mock_data = pd.DataFrame(
        {
            "ticker": ["AAPL", "TSLA"],
            "text": ["Sample text 1", "Sample text 2"],
            "flair": ["bullish", "bearish"],
            "timestamp": ["2022-01-01", "2022-01-02"],
        }
    )

    mock_path = (
        Path("data/processed") / "preprocessed_data.csv"
    )  # point to the actual CSV
    with patch("pandas.read_csv", return_value=mock_data):
        dataset = MyDataset(mock_path)

    assert len(dataset) == 2
    assert dataset[0] == {
        "ticker": "AAPL",
        "text": "Sample text 1",
        "flair": "bullish",
        "timestamp": "2022-01-01",
    }


def test_my_dataset_file_not_found():
    """Test MyDataset initialization with missing file."""
    mock_path = Path("data/processed")

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            MyDataset(mock_path)


def test_preprocess_function():
    """Test the preprocess function."""
    raw_data_path = Path("data/raw")
    output_folder = Path("data/processed")

    mock_json_path = raw_data_path / "saps.json"
    mock_data = {
        "RobinHoodPennyStocks": {
            "raw": {
                "postData": [
                    {
                        "ticker": "AAPL",
                        "title": "Sample",
                        "text": "Test Text",
                        "flair": "bullish",
                        "timestamp": "2022-01-01",
                    },
                    {
                        "ticker": "TSLA",
                        "title": "Sample",
                        "text": "Another Text",
                        "flair": "bearish",
                        "timestamp": "2022-01-02",
                    },
                ]
            }
        }
    }

    expected_df = pd.DataFrame(
        {
            "ticker": ["AAPL", "TSLA"],
            "title": ["Sample", "Sample"],
            "text": ["test text", "another text"],
            "flair": ["bullish", "bearish"],
            "timestamp": ["2022-01-01", "2022-01-02"],
        }
    )

    def mock_exists(p):
        return str(p) == str(mock_json_path)

    with patch("pathlib.Path.exists", new=mock_exists):
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                preprocess(raw_data_path, output_folder)

                mock_to_csv.assert_called_once()


def test_preprocess_file_not_found():
    """Test the preprocess function with missing json file."""
    raw_data_path = Path("data/raw")
    output_folder = Path("data/processed")

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            preprocess(raw_data_path, output_folder)


def test_my_dataset_getitem_out_of_bounds():
    """Test __getitem__ with an out-of-bounds index."""
    mock_data = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "text": ["Sample text"],
            "flair": ["bullish"],
            "timestamp": ["2022-01-01"],
        }
    )

    mock_path = Path("data/processed")
    with patch("pandas.read_csv", return_value=mock_data):
        dataset = MyDataset(mock_path)

    with pytest.raises(IndexError):
        _ = dataset[10]


if __name__ == "__main__":
    pytest.main()
