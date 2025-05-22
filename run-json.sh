#!/bin/bash

# Check if a transcript is provided as an argument
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 \"<transcript_text_or_file>\" [output_json_path]"
    exit 1
fi

TRANSCRIPT="$1"
JSON_FILE_PATH=${2:-output/results.json}  # Default to output/results.json if not provided

# Create the output directory if it doesn't exist
mkdir -p $(dirname "$JSON_FILE_PATH")

# Build the Docker image
echo "Building Docker image..."
docker build -t speech-analysis .

# Check if transcript is a file path
if [ -f "$TRANSCRIPT" ]; then
    echo "Passing transcript file to Docker: $TRANSCRIPT"
    FILENAME=$(basename "$TRANSCRIPT")
    docker run --rm -v "$(pwd)/$TRANSCRIPT":/app/$FILENAME \
        --entrypoint python \
        speech-analysis analysis3-json.py "/app/$FILENAME" > "$JSON_FILE_PATH"
else
    echo "Passing transcript text to Docker"
    docker run --rm \
        --entrypoint python \
        speech-analysis analysis3-json.py "$TRANSCRIPT" > "$JSON_FILE_PATH"
fi

# Check if the output file exists and is non-empty
if [ -s "$JSON_FILE_PATH" ]; then
    echo "Analysis complete! JSON results saved to $JSON_FILE_PATH"
    
    # Print a prettified version of the JSON if jq is installed
    if command -v jq &> /dev/null; then
        echo "===== JSON PREVIEW ====="
        jq . "$JSON_FILE_PATH" || cat "$JSON_FILE_PATH"
        echo "======================="
    else
        # Just show the first few lines of the JSON
        echo "===== JSON PREVIEW ====="
        head -20 "$JSON_FILE_PATH"
        echo "[...]"
        echo "======================="
    fi
else
    echo "Error: No output generated or file is empty."
    exit 1
fi
