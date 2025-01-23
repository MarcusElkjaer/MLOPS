import random

from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def test_get_last_month_posts(self) -> None:
        """A task that simulates a user visiting a random item URL of the FastAPI app."""
        self.client.get("/get_posts?search_term=GameStop&subreddit=wallstreetbets")

    @task(4)
    def test_analyze_sentiment(self) -> None: 
        """A task that simulates a user visiting a random item URL of the FastAPI app."""
        self.client.get("/analyze_sentiment?text=I am happy")