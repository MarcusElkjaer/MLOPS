# Welcome to Reddit Forecast Documentation

Welcome to the **Reddit Forecast** project documentation. This site provides comprehensive information about the project's API, data processing, and model details.

## 📚 Table of Contents

- [API Reference](my_api.md)
- [Data](data.md)
- [Train](train.md)
- [Model](model.md)
- [How to run](how_to_run.md)

## 🛠️ About the Project

Reddit Forecast is a tool designed to analyze Reddit posts, perform sentiment analysis, and provide insights into stock discussions and sentiments.


# Welcome to Reddit Forecast Documentation

Welcome to the **Reddit Forecast** project documentation. This site provides comprehensive information about the project's API, data processing, and model details.

## 📚 Table of Contents

- [API Reference](my_api.md)
- [Data](data.md)
- [Train](train.md)
- [Model](model.md)
- [How to run](how_to_run.md)

## 🛠️ About the Project

Reddit Forecast is a tool designed to analyze Reddit posts, perform sentiment analysis, and provide insights into stock discussions and sentiments.

## 🤝 People Who Helped

This project is a collaborative effort. Here are the people who were a part of the project

- **[Martin S. Jespersen]**
- **[Lucas Sylvester]**
- **[Lucas 2]**
- **[Marcus Elkjær]**
- **[Sebastian Wulf Andersen]**



## ✨ Features

- **Sentiment Analysis**: Analyze the sentiment of Reddit discussions related to stock tickers.
- **Data Visualization**: Generate insightful plots for class distribution and other metrics.
- **Custom API**: Query sentiment scores and retrieve processed data programmatically.
- **Scalable Model Training**: Leverage preprocessed data to train machine learning models.

## 📂 Project Structure

```
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
|       ├── otherMDFiles.md
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
|   └── test_train.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
