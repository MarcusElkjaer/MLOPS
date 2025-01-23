# Data Module Overview

This module includes data loading and preprocessing logic for Reddit ticker data.

## ⚡ Key Functions

- **preprocess(raw_data_path, output_folder)**
  Checks for a JSON file, reads data, and outputs processed artifacts, logs, and metrics.

- **dataset_statistics(processed_path)**
  Examines data shape, distribution, unique tickers, and writes a stats report in Markdown along with a class distribution plot.
