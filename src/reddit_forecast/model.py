from typing import Tuple
import pandas as pd
from transformers import pipeline


# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Analyze sentiment of the given text."""
    if pd.isnull(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return None, None
    try:
        result = sentiment_analyzer(text[:512])[0]
        return result['label'], result['score']
    except Exception as e:
        return None, None


def apply_sentiment_analysis(input_path: str, output_path: str) -> None:
    """Apply sentiment analysis to preprocessed data."""
    df = pd.read_csv(input_path)

    # Drop rows with missing or invalid text
    df = df.dropna(subset=["text"])
    df = df[df["text"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

    # Apply sentiment analysis
    df[["sentiment", "sentiment_score"]] = df["text"].apply(
        lambda text: pd.Series(analyze_sentiment(text))
    )

    # Save cleaned and processed data
    df.to_csv(output_path, index=False)
    print(f"Sentiment analysis applied and saved to {output_path}")



if __name__ == "__main__":
    input_path = "data/processed/preprocessed_data.csv"
    output_path = "models/processed_posts_with_sentiment.csv"
    apply_sentiment_analysis(input_path, output_path)
