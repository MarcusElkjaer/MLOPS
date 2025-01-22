import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from reddit_forecast.model import (
    analyze_sentiment_batch,
)
import os


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping test in Github Actions due to limited resources",
)
def test_analyze_sentiment_batch():
    """Test analyze_sentiment_batch function with valid texts."""
    mock_pipeline = MagicMock(
        return_value=[
            {"label": "POSITIVE", "score": 0.999},
            {"label": "NEGATIVE", "score": 0.888},
        ]
    )
    with patch("reddit_forecast.model.sentiment_analyzer", mock_pipeline):
        texts = ["This is a positive example.", "This is a negative example."]
        results = analyze_sentiment_batch(texts)
        assert len(results) == 2
        assert results[0] == ("POSITIVE", pytest.approx(0.999, abs=0.01))
        assert results[1] == ("NEGATIVE", pytest.approx(0.888, abs=0.01))


def test_analyze_sentiment_batch_empty_texts():
    """Test analyze_sentiment_batch with empty and invalid texts."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = []
    with patch("reddit_forecast.model.pipeline", return_value=mock_pipeline):
        texts = ["", None, "   "]
        results = analyze_sentiment_batch(texts)
        # Since the pipeline will process empty strings as None results
        assert results == [(None, None), (None, None), (None, None)]


def test_apply_sentiment_analysis():
    """Test apply_sentiment_analysis function."""
    raw_data = [
        {
            "ticker": "AAPL",
            "title": "Great stock!",
            "text": "Apple is doing great!",
            "flair": "Discussion",
            "timestamp": "2025-01-01T12:00:00",
        },
        {
            "ticker": "MSFT",
            "title": "Terrible stock!",
            "text": "Microsoft is struggling.",
            "flair": "Opinion",
            "timestamp": "2025-01-02T12:00:00",
        },
    ]
    input_df = pd.DataFrame(raw_data)
    input_path = "input.csv"
    output_path = "output.csv"
    input_df.to_csv(input_path, index=False)

    mock_analyze_sentiment_batch = MagicMock(
        return_value=[
            ("POSITIVE", 0.99),
            ("NEGATIVE", 0.88),
        ]
    )
    with patch(
        "reddit_forecast.model.analyze_sentiment_batch", mock_analyze_sentiment_batch
    ):
        from reddit_forecast.model import run_sentiment_analysis

        run_sentiment_analysis(input_path, output_path)

        result_df = pd.read_csv(output_path)
        assert "sentiment" in result_df.columns
        assert "sentiment_score" in result_df.columns
        assert result_df.loc[0, "sentiment"] == "POSITIVE"
        assert result_df.loc[1, "sentiment"] == "NEGATIVE"
        assert result_df.loc[0, "sentiment_score"] == pytest.approx(0.99, abs=0.01)
        assert result_df.loc[1, "sentiment_score"] == pytest.approx(0.88, abs=0.01)
