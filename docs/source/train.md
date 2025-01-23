# Train documentation
This project implements a sentiment regression model using `DistilBERT` for analyzing and predicting sentiment scores from a dataset of labeled sentences. The implementation leverages `PyTorch`, `PyTorch Lightning`, `Transformers`, and `Optuna` for hyperparameter tuning. The data set that we are looking at is reddit penny stocks found on kaggle.
---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
---

## Overview

This pipeline includes the following features:
- Data preprocessing and tokenization using `AutoTokenizer` from the Hugging Face Transformers library.
- Sentiment regression model built on `DistilBERT`.
- Fine-tuning using `PyTorch Lightning`.
- Automated hyperparameter tuning with `Optuna`.
- Configurable via `Hydra` to maintain modular and scalable configurations.

---

## Dataset

The dataset is provided as a CSV file with the following structure:

| Sentence                       | Sentiment  |
|--------------------------------|------------|
| "The stock is performing well" | positive   |
| "The market looks bad today"   | negative   |
| "I'm neutral about this move"  | neutral    |

- **File Location**: `data/raw/data.csv`
- **Columns**:
  - `Sentence`: The text input.
  - `Sentiment`: The sentiment label (`positive`, `negative`, `neutral`).

---

## Model Architecture

### Sentiment Regression Model
- **Base Model**: `DistilBERT`
- **Tokenization**: Uses `AutoTokenizer` from Hugging Face to preprocess the text.
- **Regression Head**: A single fully connected layer on top of the [CLS] token representation to predict a continuous sentiment score.
- **Loss Function**: Mean Squared Error (MSE).

---

## Training

### Key Components
- **Training Split**: The dataset is split into training and validation sets based on `train_split` in the configuration.
- **Optimizer**: `AdamW` with configurable learning rate and L2 regularization.
- **Trainer**: `PyTorch Lightning` handles the training and validation loops.

### Training Steps
1. **Load Dataset**: Read the CSV file and preprocess sentences using the tokenizer.
2. **Dataset Splitting**: Split the dataset into training and validation sets.
3. **Model Training**: Train the model using the configured optimizer and loss function.

---

## Hyperparameter Optimization

- **Framework**: `Optuna` is used for hyperparameter tuning.
- **Optimization Goals**: Minimize the validation loss.
- **Tunable Parameters**:
  - Learning Rate (`lr_min` to `lr_max`).
  - L2 Regularization (`l2_min` to `l2_max`).
  - Batch Size.
- **Number of Trials**: `n_trials` as specified in the configuration.
- **Storage**: Results are stored in `optuna_study.db` using SQLite.

### Best Hyperparameters Example:
```text
Best trial:
  Value: 0.01234
  Params:
    learning_rate: 6.67e-05
    l2: 0.000695
    batch_size: 48
