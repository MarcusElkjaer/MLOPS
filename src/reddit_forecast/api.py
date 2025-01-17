from fastapi import FastAPI
from reddit_forecast.model import analyze_sentiment_batch
from http import HTTPStatus
import praw
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the Reddit client
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

def get_last_month_posts(subreddit_name: str, search_term: str):
    subreddit = reddit.subreddit(subreddit_name)
    one_month_ago = datetime.now() - timedelta(days=30)
    posts = []

    for submission in subreddit.search(search_term, sort='new', time_filter='month'):
        if datetime.fromtimestamp(submission.created_utc) >= one_month_ago:
            posts.append({
                'title': submission.title,
                'url': submission.url,
                'created_date': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d'),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'selftext': submission.selftext  # Fetch the text body of the post
            })
    
    # Sort posts by created_date
    posts.sort(key=lambda x: x['created_date'])
    return posts

def analyze_posts_sentiment(posts):
    texts = [post['title'] + " " + post['selftext'] for post in posts]
    sentiments = analyze_sentiment_batch(texts)
    for post, sentiment in zip(posts, sentiments):
        post['sentiment'] = sentiment
    return posts

app = FastAPI()

@app.get("/")
def read_root():
    return HTTPStatus.OK

@app.get("/analyze_sentiment")
def analyze_sentiment(text: str):
    return analyze_sentiment_batch([text])

@app.get("/get_last_month_posts")
def get_last_month_posts_endpoint(search_term: str, subreddit: str = "wallstreetbets"):
    posts = get_last_month_posts(subreddit, search_term)
    posts_with_sentiment = analyze_posts_sentiment(posts)
    return posts_with_sentiment