#!/bin/bash

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1

echo "Building Docker image with model precaching (this may take a few minutes)..."

# Build with no cache if requested
if [ "$1" == "--no-cache" ]; then
    echo "Building without cache..."
    docker build --progress=plain --no-cache -t speech-analysis .
else
    echo "Building with cache..."
    docker build --progress=plain -t speech-analysis .
fi

echo "Build complete! The next runs will be much faster because models are now cached in the image."

