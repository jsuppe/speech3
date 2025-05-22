#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t speech-analysis .

# Run an interactive shell in the container
echo "Starting interactive shell in container..."
docker run --rm -it --entrypoint /bin/bash speech-analysis

echo "Container exited."
