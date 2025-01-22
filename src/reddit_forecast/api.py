from fastapi import FastAPI
from fastapi.testclient import TestClient
from reddit_forecast.model import analyze_sentiment_batch
from http import HTTPStatus
import praw
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from collections import defaultdict

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# Load environment variables from .env file
load_dotenv()
app = FastAPI()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# You can add additional URLs to this list, for example, the frontend's production domain, or other frontends.
allowed_origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["X-Requested-With", "Content-Type"],
)

test_client = TestClient(app)



# Initialize the Reddit client
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent= "reddit_forecast"
)

def get_posts(subreddit_name: str, search_term: str, time = datetime.now() - timedelta(days=30)):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.search(search_term, sort='new', time_filter='month'):
        if datetime.fromtimestamp(submission.created_utc) >= time:
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

def calculate_average_sentiment(posts):
    date_sentiment_map = defaultdict(list)
    for post in posts:
        sentiment_score = post['sentiment'][1]
        sign = post['sentiment'][0]
        if sign == 'NEGATIVE':
            sentiment_score *= -1
        date_sentiment_map[post['created_date']].append(sentiment_score)
    
    average_sentiment_per_day = [
        {'date': date, 'average_sentiment': sum(scores) / len(scores)}
        for date, scores in date_sentiment_map.items()
    ]
    
    # Sort by date
    average_sentiment_per_day.sort(key=lambda x: x['date'])
    return average_sentiment_per_day

# @app.get("/")
# def read_root():
#     return HTTPStatus.OK

@app.get("/analyze_sentiment")
def analyze_sentiment(text: str):
    return analyze_sentiment_batch([text])[0]

@app.get("/get_posts")
def get_posts_endpoint(search_term: str, subreddit: str = "wallstreetbets"):
    posts = get_posts(subreddit, search_term)
    posts_with_sentiment = analyze_posts_sentiment(posts)
    return posts_with_sentiment

@app.get("/get_average_sentiment")
def get_average_sentiment_endpoint(search_term: str, subreddit: str = "wallstreetbets"):
    posts = get_posts(subreddit, search_term)
    posts_with_sentiment = analyze_posts_sentiment(posts)
    average_sentiment = calculate_average_sentiment(posts_with_sentiment)
    return average_sentiment


#code to server the react app in frontend/dist/index.html
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
templates = Jinja2Templates(directory="frontend/dist")
@app.get("/", include_in_schema=False)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# run with: uvicorn reddit_forecast.api:app --reload --port 8000