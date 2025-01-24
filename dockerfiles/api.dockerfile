# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

# Install build-essential and Node.js
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc nodejs npm && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY src src/
COPY data data/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install Python dependencies
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt
#RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Copy frontend files and install Node.js dependencies
COPY frontend frontend/
WORKDIR /frontend
RUN npm install
RUN npm run build

# Run the frontend and backend servers
WORKDIR /
EXPOSE 8000

RUN python -m reddit_forecast.data_drift

# Start both the backend and frontend and generate datadrift report
CMD ["sh", "-c", "uvicorn reddit_forecast.api:app --host 0.0.0.0 --port 8000"]

