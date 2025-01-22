from pathlib import Path
import json
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional
import typer
import torch
import logging
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        :param csv_path: Full path to the CSV file
                         (e.g. data/processed/preprocessed_data.csv)
        :param transform: Optional transform to apply to each row dictionary
        """
        self.csv_path = Path(csv_path)  # make sure it's a Path object
        self.transform = transform

        # Check that it's actually a file, not a directory
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"{self.csv_path} not found or is not a file.")

        # Read the entire CSV at once
        self.df = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Check bounds
        if idx >= len(self.df):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.df)}"
            )

        # Extract the row as a dictionary of {column_name: value}
        row_dict = self.df.iloc[idx].to_dict()

        # Apply any transform if needed
        if self.transform:
            row_dict = self.transform(row_dict)

        return row_dict


def dataset_statistics(processed_path: Path):
    """Generate and save statistics about the dataset."""
    preprocessed_file = processed_path / "preprocessed_data.csv"
    if not preprocessed_file.exists():
        raise FileNotFoundError(f"{preprocessed_file} not found.")

    data = pd.read_csv(preprocessed_file)
    # Basic statistics
    num_samples = len(data)
    num_unique_tickers = data["ticker"].nunique()
    class_distribution = data["flair"].value_counts()

    # Save class distribution plot
    plt.figure(figsize=(8, 6))
    class_distribution.plot(kind="bar")
    plt.title("Class Distribution")
    plt.savefig("class_distribution.png")
    plt.close()

    # Write statistics to a markdown file
    output_file = processed_path / "statistics_report.md"
    with open(output_file, "w") as report:
        report.write(f"# Dataset Statistics\n\n")
        report.write(f"Number of samples: {num_samples}\n")
        report.write(f"Number of unique tickers: {num_unique_tickers}\n\n")
        report.write("## Class Distribution\n\n")
        report.write(class_distribution.to_markdown())


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    logger.info("Preprocessing data...")
    json_path = raw_data_path / "saps.json"
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    # Load the raw JSON
    logger.info(f"Loading data from {json_path}")
    with open(json_path, "r") as file:
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
    from pathlib import Path
    from google.cloud import storage

    # Run DVC pull
    try:
        print("Running DVC pull to fetch data...")
        subprocess.run(["dvc", "pull", "--force"], check=True)
        print("DVC pull completed.")
    except:
        print("DVC pull failed.")
    
    dataset = "justinmiller/reddit-pennystock-data"
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
            check=True,
        )
    else:
        print("Dataset already downloaded.")

    # Extract dataset if not already extracted
    extracted_folder = raw_data_path / dataset.split("/")[1]
    if not extracted_folder.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_data_path)
    else:
        print("Dataset already extracted.")

    # Preprocess data
    if not (processed_path / "preprocessed_data.csv").exists():
        preprocess(raw_data_path, processed_path)
    else:
        print(f"Preprocessed data already exists in {processed_path}.")

    # Download additional dataset if not already downloaded
    additional_dataset = "sbhatti/financial-sentiment-analysis"
    additional_zip_path = raw_data_path / f'{additional_dataset.split("/")[1]}.zip'

    if not additional_zip_path.exists():
        print("Downloading additional dataset...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                additional_dataset,
                "-p",
                str(raw_data_path),
            ],
            check=True,
        )
    else:
        print("Additional dataset already downloaded.")

    # Extract additional dataset if not already extracted
    additional_extracted_folder = raw_data_path / additional_dataset.split("/")[1]
    if not additional_extracted_folder.exists():
        print("Extracting additional dataset...")
        with zipfile.ZipFile(additional_zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_data_path)
    else:
        print("Additional dataset already extracted.")
