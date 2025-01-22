#!/bin/bash

# Configuration variables
REGISTRY="europe-west4-docker.pkg.dev"
PROJECT_ID="mlops-448109"
REPOSITORY="mlops-artifact-registry"
IMAGE_NAME="reddit_forecast_model" # Replace with your image name
TAG="latest"        # Replace with your desired tag, e.g., latest
LOCAL_PORT=8000     # Port on your local machine
CONTAINER_PORT=8000 # Port the application listens to in the container

# Authenticate Docker with Artifact Registry
echo "Authenticating Docker with Google Artifact Registry..."
gcloud auth configure-docker $REGISTRY

# Pull the latest image
echo "Pulling the latest image from $REGISTRY/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$TAG..."
docker pull $REGISTRY/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$TAG

# Run the container
echo "Running the container..."
CONTAINER_ID=$(docker run -d -p $LOCAL_PORT:$CONTAINER_PORT $REGISTRY/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$TAG)

# Wait for the container to be ready
echo "Waiting for the container to bind to port $LOCAL_PORT..."
MAX_RETRIES=30
RETRY_COUNT=0
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
  # Check if the port is being used
  if lsof -i TCP:$LOCAL_PORT | grep LISTEN > /dev/null; then
    echo "Application is up and running at http://localhost:$LOCAL_PORT"
    break
  fi
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "Waiting for the application... (retry $RETRY_COUNT/$MAX_RETRIES)"
  sleep 2
done

# If the app doesn't start, exit with an error
if [[ $RETRY_COUNT -eq $MAX_RETRIES ]]; then
  echo "Application did not start within the expected time. Check the container logs for issues."
  echo "Fetching logs for container $CONTAINER_ID:"
  docker logs $CONTAINER_ID
  exit 1
fi

# Verify the container is running
echo "Checking running containers..."
docker ps

echo "Container is running. Access it at http://localhost:$LOCAL_PORT"
