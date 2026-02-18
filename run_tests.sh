#!/bin/bash
# SpeakFit API Test Runner
# Run after every change to catch performance regressions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  SpeakFit API Test Suite"
echo "=========================================="

# Check if API is running
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Local API not running at localhost:8000${NC}"
    echo "Starting API..."
    nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /tmp/speechscore_api.log 2>&1 &
    sleep 3
fi

# Install test dependencies if needed
pip install -q pytest httpx 2>/dev/null || true

echo ""
echo "Running local benchmarks..."
echo "------------------------------------------"

# Run local tests
if pytest tests/test_api_benchmarks.py -v --tb=short; then
    echo -e "\n${GREEN}✓ Local benchmarks passed${NC}"
else
    echo -e "\n${RED}✗ Local benchmarks failed${NC}"
    exit 1
fi

# Optionally run production tests
if [ "$1" == "--prod" ]; then
    echo ""
    echo "Running production benchmarks..."
    echo "------------------------------------------"
    API_BASE_URL=https://api.speakfit.app pytest tests/test_api_production.py -v --tb=short
fi

echo ""
echo "=========================================="
echo -e "${GREEN}  All tests passed!${NC}"
echo "=========================================="
