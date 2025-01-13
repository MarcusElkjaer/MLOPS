from pathlib import Path

from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    #download data
    import os
    import zipfile
    import kaggle
    
    dataset = 'justinmiller/reddit-pennystock-data'
    download_path = 'data/raw'

    #check if the data is already downloaded
    os.system(f'kaggle datasets download -d {dataset} -p {download_path}')
    
    zip_path = f'{download_path}/{dataset.split("/")[1]}.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    
    #preprocess data
    