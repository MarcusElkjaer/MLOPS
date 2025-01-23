# API Reference

This section provides detailed information about the API endpoints available in the Reddit Forecast project.

## Endpoints

### Analyze Sentiment

**Endpoint:** `/analyze_sentiment`

**Method:** `GET`

**Description:** Analyzes the sentiment of the provided text.

**Parameters:**
- `text` (str): The text to analyze.

**Response:**
- `200 OK`: Returns the sentiment of the text.

**Example Request:**
```http
GET /analyze_sentiment?text=I am happy
```
**Example Reponse**
```

```

### Get Posts

**Endpoint:** `/get_posts`

**Method:** `GET`

**Description:** Retrieves posts from a specified subreddit based on a search term and analyzes their sentiment.

**Parameters:**
- `search_term` (str): The term to search for in the subreddit.
- `subreddit` (str, optional): The subreddit to search in. Default is `wallstreetbets`.

**Response:**
- `200 OK`: Returns a list of posts with their sentiment analysis.

**Example Request:**
```http
GET /get_posts?search_term=GameStop&subreddit=wallstreetbets
```
**Example Response**
```
[
  {
    "title": "GameStop to the moon!",
    "url": "https://www.reddit.com/r/wallstreetbets/comments/...",
    "created_date": "2023-10-01",
    "score": 1234,
    "num_comments": 567,
    "selftext": "GameStop is going to skyrocket!",
    "sentiment": "POSITIVE"
  },
  ...
]
```

### Get Average Sentiment

**Endpoint:** `/get_average_sentiment`

**Method:** `GET`

**Description:** Retrieves posts from a specified subreddit based on a search term and calculates the average sentiment per day.

**Parameters:**
- `search_term` (str): The term to search for in the subreddit.
- `subreddit` (str, optional): The subreddit to search in. Default is `wallstreetbets`.

**Response:**
- `200 OK`: Returns the average sentiment per day.

**Example Request:**
```http
GET /get_average_sentiment?search_term=GameStop&subreddit=wallstreetbets
```
**Example Response**
```
[
  {
    "date": "2023-10-01",
    "average_sentiment": 0.75
  },
  ...
]
```

### Get Report

**Endpoint:** `/report`

**Method:** `GET`

**Description:** Generates and returns the data drift report.

**Response:**
- `200 OK`: Returns the HTML content of the data drift report.

**Example Request:**
```http
GET /report
```