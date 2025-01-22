import pytest
import torch
from pytest import approx
from unittest.mock import MagicMock, patch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertModel
from src.reddit_forecast.train import SentimentDataset, SentimentRegressionModel, seed_randoms, objective

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode_plus.return_value = {
        "input_ids": torch.tensor([[101, 2057, 2293, 21710, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
    }
    return tokenizer


def test_sentiment_dataset(mock_tokenizer):
    mock_cfg = MagicMock()
    mock_cfg.train.max_length = 128
    mock_file_path = "tests/mock_data.csv"

    fake_data = pd.DataFrame({
        "Sentence": ["I love this!", "This is terrible!", "It's okay."],
        "Sentiment": ["positive", "negative", "neutral"]
    })

    with patch("pandas.read_csv", return_value=fake_data):
        dataset = SentimentDataset(mock_file_path, mock_tokenizer, mock_cfg)

        assert len(dataset) == 3  # Check if the mocked data length matches
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "label" in sample




def test_sentiment_regression_model():
    model_name = "distilbert-base-uncased"
    model = SentimentRegressionModel(model_name, learning_rate=1e-7, l2=1e-6)
    assert isinstance(model.model, DistilBertModel)  # Check for specific model type



def test_training_step():
    model_name = "distilbert-base-uncased"
    model = SentimentRegressionModel(model_name, learning_rate=1e-7, l2=1e-6)
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones((2, 128)),
        "label": torch.tensor([1.0, 0.0])
    }
    loss = model.training_step(batch, 0)
    assert loss > 0


def test_validation_step():
    model_name = "distilbert-base-uncased"
    model = SentimentRegressionModel(model_name, learning_rate=1e-7, l2=1e-6)
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones((2, 128)),
        "label": torch.tensor([1.0, 0.0])
    }
    loss = model.validation_step(batch, 0)
    assert loss > 0


def test_seed_randoms():
    mock_cfg = MagicMock()
    mock_cfg.train.seed = 42
    seed_randoms(mock_cfg)
    assert torch.initial_seed() == 42


def test_objective():
    trial = MagicMock()
    trial.suggest_float.side_effect = [1e-7, 1e-6]
    trial.suggest_categorical.return_value = 32
    mock_cfg = MagicMock()
    mock_cfg.train.lr_min = 1e-7
    mock_cfg.train.lr_max = 1e-3
    mock_cfg.train.l2_min = 1e-6
    mock_cfg.train.l2_max = 1e-3
    mock_cfg.train.batch_size = 32
    mock_cfg.train.optuna_epoch = 3

    train_dataset = MagicMock()
    train_dataset.__len__.return_value = 100  # Ensure dataset length is positive

    val_dataset = MagicMock()
    val_dataset.__len__.return_value = 50  # Ensure dataset length is positive
    
    with patch("pytorch_lightning.Trainer.fit") as mock_fit:
        mock_fit.return_value = None
        with patch("pytorch_lightning.Trainer.callback_metrics", {"val_loss": torch.tensor(0.1)}):
            result = objective(trial, train_dataset, val_dataset, mock_cfg)

    assert result == approx(0.1, rel=1e-3)  # Ensure the returned loss is as expected


