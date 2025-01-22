# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

# Install build-essential and Node.js
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc nodejs npm && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install Python dependencies
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
#RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Copy frontend files and install Node.js dependencies
COPY frontend frontend/
WORKDIR /frontend
RUN npm install

# Run the frontend and backend servers
WORKDIR /
EXPOSE 5173 8000

# Start both the backend and frontend
CMD ["sh", "-c", "uvicorn reddit_forecast.api:app --host 0.0.0.0 --port 8000:8080 & npm run --prefix frontend dev -- --host"]


