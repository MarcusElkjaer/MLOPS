from typing import Tuple
import pandas as pd
from transformers import pipeline
import logging
from typing import Tuple, Optional
import hydra
import os
from hydra.utils import to_absolute_path


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment_batch(
    texts: list[str],
) -> list[Tuple[Optional[str], Optional[float]]]:
    """Analyze sentiment for a batch of texts."""
    results = []
    try:
        responses = sentiment_analyzer(texts, truncation=True, padding=True)
        for response in responses:
            results.append((response["label"], response["score"]))
    except Exception as e:
        logger.error(f"Failed to analyze sentiment for batch. Error: {e}")
        results = [(None, None)] * len(texts)
    return results


@hydra.main(config_path="../../configs", config_name="config.yaml")
def apply_sentiment_analysis(cfg) -> None:
    """
    Apply sentiment analysis to preprocessed data.
    Assumes cfg.model.input_path, cfg.model.output_path, cfg.model.batch_size exist.
    """
    # Read input CSV from the specified path
    # (If your config has relative paths, you may need to convert them to absolute with to_absolute_path)
    input_path = to_absolute_path(cfg.model.input_path)
    output_path = (
        cfg.model.output_path
    )  # This can remain relative to Hydra’s run dir if you like

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["text"])
    df = df[df["text"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

    # Perform sentiment analysis in batches
    batch_size = cfg.model.batch_size
    sentiments = []
    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].iloc[i : i + batch_size].tolist()
        sentiments.extend(analyze_sentiment_batch(batch_texts))

    df[["sentiment", "sentiment_score"]] = pd.DataFrame(sentiments)

    # **Create the directory if it doesn't exist**
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    input_path = "data/processed/preprocessed_data.csv"
    output_path = "models/processed_posts_with_sentiment.csv"
    apply_sentiment_analysis()
