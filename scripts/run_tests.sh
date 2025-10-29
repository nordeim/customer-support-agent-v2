#!/bin/bash

# Test runner script with comprehensive checks
# Usage: ./scripts/run_tests.sh [options]

pip install pytest pytest-asyncio pytest-cov

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
COVERAGE=true
VERBOSE=false
FAIL_UNDER=80
MARKERS=""
WARNINGS="error"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --fail-under)
            FAIL_UNDER="$2"
            shift 2
            ;;
        --mark|-m)
            MARKERS="-m $2"
            shift 2
            ;;
        --warnings)
            WARNINGS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --no-coverage         Disable coverage reporting"
            echo "  --verbose, -v         Enable verbose output"
            echo "  --fail-under <num>    Minimum coverage percentage (default: 80)"
            echo "  --mark, -m <marker>   Run only tests with specific marker"
            echo "  --warnings <mode>     Warning mode: error|default|ignore"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests with coverage"
            echo "  $0 -m unit            # Run only unit tests"
            echo "  $0 -m \"not slow\"      # Skip slow tests"
            echo "  $0 --no-coverage -v   # Run without coverage, verbose"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Install with: pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Set testing environment
export TESTING=true
export ENVIRONMENT=testing
export DEBUG=true
export ENABLE_TELEMETRY=false

# Build pytest command
PYTEST_CMD="pytest"

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage options
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=app --cov-report=term-missing --cov-report=html --cov-fail-under=$FAIL_UNDER"
fi

# Add markers
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

# Add warnings mode
PYTEST_CMD="$PYTEST_CMD -W $WARNINGS"

# Add color and show local variables on failure
PYTEST_CMD="$PYTEST_CMD --color=yes --tb=short"

# Run tests
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo ""

if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
