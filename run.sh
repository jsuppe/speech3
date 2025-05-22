#!/bin/bash

SCRIPT=analysis3.py

# Check if a transcript is provided as an argument
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 \"<transcript_text_or_file>\" [results_file_path]"
    exit 1
fi

TRANSCRIPT="$1"
RESULTS_FILE_PATH=${2:-output/results.txt}  # Default to output/results.txt if not provided

# Create the output directory if it doesn't exist
mkdir -p $(dirname "$RESULTS_FILE_PATH")

# Delete the docker container if it exists
docker rm -f speech-analysis 2>/dev/null

# Build the Docker image
echo "Building Docker image..."
docker build -t speech-analysis .

# Add a note about performance
echo "Running analysis (first run might take longer while loading models)..."

# Check if transcript is a file path
if [ -f "$TRANSCRIPT" ]; then
    echo "Passing transcript file to Docker: $TRANSCRIPT"
    FILENAME=$(basename "$TRANSCRIPT")
    docker run --rm -v "$(pwd)/$(dirname "$RESULTS_FILE_PATH")":/app/output \
        -v "$(pwd)/$TRANSCRIPT":/app/$FILENAME \
        -e RESULTS_FILE_PATH=/app/output/$(basename "$RESULTS_FILE_PATH") \
        speech-analysis "/app/$FILENAME" --output="/app/output/$(basename "$RESULTS_FILE_PATH")"
else
    echo "Passing transcript text to Docker"
    docker run --rm -v "$(pwd)/$(dirname "$RESULTS_FILE_PATH")":/app/output \
        -e RESULTS_FILE_PATH=/app/output/$(basename "$RESULTS_FILE_PATH") \
        speech-analysis "$TRANSCRIPT" --output="/app/output/$(basename "$RESULTS_FILE_PATH")"
fi

# List the contents of the output directory to verify results
ls -l "$(dirname "$RESULTS_FILE_PATH")"
echo "Analysis complete! Results saved to $RESULTS_FILE_PATH"

# Display the contents of the output file
echo "===== OUTPUT FILE CONTENTS ====="
cat "$RESULTS_FILE_PATH"
echo "==============================="
