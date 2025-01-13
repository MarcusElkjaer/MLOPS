import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from reddit_forecast.model import (
    analyze_sentiment, 
)
            
def test_analyze_sentiment():
    """Test analyze_sentiment function."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9998446702957153}]
    with patch("reddit_forecast.model.pipeline", return_value=mock_pipeline):
        text = "This is a positive example."
        result = analyze_sentiment(text)
        # Use pytest.approx for floating-point comparison
        assert result[0] == "POSITIVE"
        assert result[1] == pytest.approx(0.95, abs=0.1)  # Allow small relative difference


def test_analyze_sentiment_empty_text():
    """Test analyze_sentiment with empty text."""
    assert analyze_sentiment("") == (None, None)
    assert analyze_sentiment(None) == (None, None)
    assert analyze_sentiment("   ") == (None, None)
    
def test_dataframe_creation():
    """Test the creation of DataFrame from raw JSON."""
    raw_data = [
        {"ticker": "AAPL", "title": "Great stock!", "text": "Apple is doing great!", "flair": "Discussion", "timestamp": "2025-01-01T12:00:00"}
    ]
    inter_keys = ["key1", "key2"]
    intra_keys = ["key3", "key4"]
    
    posts_df = pd.DataFrame(raw_data, columns=["ticker", "title", "text", "flair", "timestamp"])
    posts_df["inter_fin_data"] = posts_df.index.map(lambda x: inter_keys[x] if x < len(inter_keys) else None)
    posts_df["intra_fin_data"] = posts_df.index.map(lambda x: intra_keys[x] if x < len(intra_keys) else None)
    
    assert "inter_fin_data" in posts_df.columns
    assert "intra_fin_data" in posts_df.columns
    assert posts_df.loc[0, "inter_fin_data"] == "key1"
    assert posts_df.loc[0, "intra_fin_data"] == "key3"




    
   
