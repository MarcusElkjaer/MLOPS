from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, TextEvals
from reddit_forecast.api import analyze_posts_sentiment, get_latest
from reddit_forecast.data import MyDataset
from datetime import datetime, timedelta
import pandas as pd

def apply_sentiment_analysis(df, title_column="title", selftext_column="selftext"):
    """
    Apply sentiment analysis to a DataFrame by combining 'title' and 'selftext',
    and create a new 'sentiment' column with the sentiment score for each post.

    Args:
        df (pd.DataFrame): The input DataFrame with columns 'title' and 'selftext'.
        title_column (str): The column name containing the post titles.
        selftext_column (str): The column name containing the post selftexts.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'sentiment' column containing sentiment scores.
    """
    # Step 1: Convert the DataFrame rows into a list of posts (as dictionaries)
    posts = df.apply(lambda row: {'title': row[title_column], 'selftext': row[selftext_column]}, axis=1).tolist()

    # Step 2: Apply sentiment analysis on all posts (assuming analyze_posts_sentiment() handles multiple posts)
    analyzed_posts = analyze_posts_sentiment(posts)

    # Step 3: Extract the sentiment scores from the analyzed posts
    sentiments = []
    for post in analyzed_posts:
        sentiment_score = post['sentiment'][1]
        sign = post['sentiment'][0]
        if sign == 'NEGATIVE':
            sentiment_score *= -1
        sentiments.append(sentiment_score)
    # Step 4: Add the sentiment scores as a new column in the DataFrame
    df['sentiment'] = sentiments
    df['combi'] = df['title'] + " " + df['selftext']

    return df


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TextEvals(column_name="combi"), TargetDriftPreset(columns=["sentiment"])])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    text_overview_report.save_html("monitoring.html")
    return text_overview_report.get_html()
def calculate_data_drift():
    df_old = MyDataset("data/processed/preprocessed_data.csv").df
    df_old.rename(columns={'text': 'selftext'}, inplace=True)
    df_old.dropna(subset=["selftext"], inplace=True)
    #df_old.dropzero(subset=["selftext"], inplace=True)
    df_old = df_old[:100]
    

    this_month = get_latest("RobinHoodPennyStocks")
    df_new = pd.DataFrame(this_month)
    df_new = apply_sentiment_analysis(df_new)
    df_old = apply_sentiment_analysis(df_old)

    run_analysis(df_old, df_new)


if __name__ == "__main__":
    calculate_data_drift()