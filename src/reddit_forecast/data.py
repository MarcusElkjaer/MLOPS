from pathlib import Path
import json
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional
import typer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
            """
            Args:
                raw_data_path: Path to the directory containing 'preprocessed_data.csv'.
            """
            self.data_path = raw_data_path
            preprocessed_file = raw_data_path / "preprocessed_data.csv"
            if not preprocessed_file.exists():
                raise FileNotFoundError(f"Could not find {preprocessed_file}")

            self.data = pd.read_csv(preprocessed_file)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Return a given sample from the dataset as a dictionary.

        Args:
            index: Index of the sample.

        Returns:
            A dictionary containing relevant fields from the dataset row.
        """
        sample = self.data.iloc[index]
        return {
            "ticker": sample["ticker"],
            "text": sample["text"],
            "flair": sample["flair"],
            "timestamp": sample["timestamp"],
        }


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    logger.info("Preprocessing data...")
    json_path = raw_data_path / "saps.json"
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    # Load the raw JSON
    logger.info(f"Loading data from {json_path}")
    with open(json_path, 'r') as file:
        raw_data = json.load(file)

    # Extract and clean the data
    posts = raw_data["RobinHoodPennyStocks"]["raw"]["postData"]
    df = pd.DataFrame(posts, columns=["ticker", "title", "text", "flair", "timestamp"])

    # Drop missing values, clean text
    df.dropna(subset=["text"], inplace=True)
    df["text"] = df["text"].str.lower().str.replace(r"[^a-z\s]", "", regex=True)

    # Save the preprocessed data
    output_path = output_folder / "preprocessed_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    import os
    import zipfile
    import subprocess

    dataset = 'justinmiller/reddit-pennystock-data'
    raw_data_path: str = Path("data/raw")
    processed_path: str = Path("data/processed")




    # Ensure directories exist
    raw_data_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_data_path / f'{dataset.split("/")[1]}.zip'

    # Download dataset if not already downloaded
    if not zip_path.exists():
        print("Downloading dataset...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_data_path)],
            check=True
        )
    else:
        print("Dataset already downloaded.")

    # Extract dataset if not already extracted
    extracted_folder = raw_data_path / dataset.split("/")[1]
    if not extracted_folder.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_path)
    else:
        print("Dataset already extracted.")

    # Preprocess data
    if not (processed_path / "preprocessed_data.csv").exists():
        preprocess(raw_data_path, processed_path)
    else:
        print(f"Preprocessed data already exists in {processed_path}.")

    # Download additional dataset if not already downloaded
    additional_dataset = 'sbhatti/financial-sentiment-analysis'
    additional_zip_path = raw_data_path / f'{additional_dataset.split("/")[1]}.zip'

    if not additional_zip_path.exists():
        print("Downloading additional dataset...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", additional_dataset, "-p", str(raw_data_path)],
            check=True
        )
    else:
        print("Additional dataset already downloaded.")

    # Extract additional dataset if not already extracted
    additional_extracted_folder = raw_data_path / additional_dataset.split("/")[1]
    if not additional_extracted_folder.exists():
        print("Extracting additional dataset...")
        with zipfile.ZipFile(additional_zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_path)
    else:
        print("Additional dataset already extracted.")