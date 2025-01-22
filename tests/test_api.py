from reddit_forecast.api import test_client
import pytest

def test_read_root():
    response = test_client.get("/")
    assert response.status_code == 200

def test_analyze_sentiment():
    response = test_client.get("/analyze_sentiment?text=I am happy")
    assert response.status_code == 200
    assert response.json()[0] == 'POSITIVE'

def test_get_last_month_posts():
    response = test_client.get("/get_posts?search_term=GameStop&subreddit=wallstreetbets")
    assert response.status_code == 200
    assert len(response.json()) > 0
    assert 'title' in response.json()[0]
    assert 'url' in response.json()[0]
    assert 'created_date' in response.json()[0]
    assert 'score' in response.json()[0]
    assert 'num_comments' in response.json()[0]
    assert 'selftext' in response.json()[0]
    assert 'sentiment' in response.json()[0]