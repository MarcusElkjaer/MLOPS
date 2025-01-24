# Data Module Overview

This module handles data loading, preprocessing, and statistics generation for Reddit ticker data. It is a critical part of the pipeline, ensuring that raw data is transformed into a usable format for analysis and machine learning tasks.

## ⚡ Key Functions

### **preprocess(raw_data_path, output_folder)**
- **Description**: Reads the JSON file containing raw data, processes it, and outputs cleaned data as CSV files.
- **Inputs**:
  - `raw_data_path (Path)`: Path to the folder containing the raw JSON file (`saps.json`).
  - `output_folder (Path)`: Path to save the processed data.
- **Outputs**:
  - `preprocessed_data.csv`: Cleaned and preprocessed data.
- **Logging**:
  - Logs the data processing steps.
- **Raises**:
  - `FileNotFoundError` if the JSON file does not exist.

### **dataset_statistics(processed_path)**
- **Description**: Analyzes the preprocessed data to compute key statistics and save a class distribution plot.
- **Inputs**:
  - `processed_path (Path)`: Path to the folder containing `preprocessed_data.csv`.
- **Outputs**:
  - `statistics_report.md`: A Markdown file with dataset statistics.
  - `class_distribution.png`: A bar plot of class distribution.
- **Raises**:
  - `FileNotFoundError` if the `preprocessed_data.csv` file does not exist.

### **MyDataset (class)**
- **Description**: A PyTorch `Dataset` class for loading and accessing preprocessed data.
- **Attributes**:
  - `csv_path (Path)`: Path to the CSV file.
  - `transform`: Optional transformation applied to the data rows.
- **Methods**:
  - `__len__`: Returns the number of rows in the dataset.
  - `__getitem__(idx)`: Retrieves a row by index and applies optional transformations.

## 📂 File Structure

```
project/
├── data/
│   ├── raw/
│   │   └── saps.json
│   ├── processed/
│   │   ├── preprocessed_data.csv
├── src/
│   └── data.py
```

## 🛠️ Usage

### Preprocessing Data
1. Place the raw data file (`saps.json`) in the `data/raw/` folder or have them downloaded with kaggle/our bucket
2. Run the preprocessing function:

```bash
python src/data.py
```

This will:
- Load raw JSON data.
- Clean and preprocess the data.
- Save the output to `data/processed/preprocessed_data.csv`.

### Generating Dataset Statistics
1. Ensure `preprocessed_data.csv` exists in `data/processed/`.
2. Call the `dataset_statistics` function:

```python
from data import dataset_statistics

processed_path = Path("data/processed")
dataset_statistics(processed_path)
```

This will:
- Compute and save statistics in `statistics_report.md`.

## 📊 Example Output

### Dataset Statistics Report

#### Sample Output from `statistics_report.md`:

```
# Dataset Statistics

Number of samples: 12345
Number of unique tickers: 678

# Dataset Statistics

Number of samples: 4800
Number of unique tickers: 592

## Class Distribution

| flair      |   count |
|:-----------|--------:|
| Discussion |     724 |
| Question   |     355 |
| Shitpost   |     311 |
| Research   |     299 |
| News       |     238 |
| Positions  |     208 |
| Rants      |      34 |
| Options    |       8 |
```
