#!/bin/bash

# Build the Docker image with no cache to ensure fresh dependencies
docker build --no-cache -t speech-analysis .

# Run a test to check if the dependencies are working correctly
docker run --rm speech-analysis -c "import sentence_transformers; print('sentence_transformers version:', sentence_transformers.__version__)"

# If the test passes, run an interactive shell for further testing
echo "If you see the sentence_transformers version above, dependencies are working correctly."
echo "Starting interactive shell for further testing..."
docker run --rm -it --entrypoint /bin/bash speech-analysis
