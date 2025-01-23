# Frontend

## How to run
Uvicorn is used to serve both the frontend and the api
It can be run simply with the following cmd, after installing requirements.
```
uvicorn reddit_forecast.api:app --reload --port 8000
```

## How to use

The frontend is available in the root "/".

Input into the field any Ticker symbol (letter combination representing a stock).
The graphs show the latest months stock price and the average sentiment for the stock each day.


## Datadrift report
To acces the data drift report go to /report.
The report compares reference data that is pulled at the time of running, with evaluation data for the model. Both datasets are from the r/RobinHoodPennyStocks subreddit.
If you wish to regenerate the report run:
```
python -m reddit_forecast.data_drift
```