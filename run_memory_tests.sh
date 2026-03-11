#!/bin/bash
# Run Memory App tests
#
# Usage:
#   ./run_memory_tests.sh           # Run all tests
#   ./run_memory_tests.sh unit      # Run unit tests only (no API)
#   ./run_memory_tests.sh api       # Run API tests only (requires running server)
#   ./run_memory_tests.sh quick     # Run quick sanity checks

set -e

cd "$(dirname "$0")"
source venv/bin/activate

export API_BASE="${API_BASE:-http://localhost:8000}"
export API_KEY="${API_KEY:-sk-5b7fd66025ead6b8731ef73b2c970f26}"

echo "============================================"
echo "  Memory App Test Suite"
echo "============================================"
echo "API: $API_BASE"
echo ""

case "${1:-all}" in
    unit)
        echo "Running unit tests (no API required)..."
        pytest tests/test_memory_analysis.py -v --tb=short
        ;;
    api)
        echo "Running API integration tests..."
        pytest tests/test_memory_api.py -v --tb=short
        ;;
    quick)
        echo "Running quick sanity checks..."
        pytest tests/test_memory_analysis.py::TestPrepositionAnalysis::test_basic_preposition_count -v
        pytest tests/test_memory_api.py::TestMemoryAPIHealth -v
        ;;
    all)
        echo "Running all tests..."
        pytest tests/test_memory_analysis.py tests/test_memory_api.py -v --tb=short
        ;;
    *)
        echo "Usage: $0 [unit|api|quick|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Tests completed!"
echo "============================================"
