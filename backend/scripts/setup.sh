#!/bin/bash

# Customer Support AI Agent - Backend Setup Script
set -e

echo "========================================="
echo "Customer Support AI Agent - Backend Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python $python_version is installed"
else
    print_error "Python $required_version or higher is required (found $python_version)"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating directories..."
mkdir -p data logs data/uploads data/chroma_db

# Copy environment file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please update .env with your configuration"
    else
        print_warning ".env.example not found"
    fi
else
    print_warning ".env file already exists"
fi

# Initialize database
print_status "Initializing database..."
python -c "from app.database import init_db; init_db()"

# Run database migrations
if [ -f "alembic.ini" ]; then
    print_status "Running database migrations..."
    alembic upgrade head
else
    print_warning "Alembic not configured, skipping migrations"
fi

# Seed sample data (optional)
read -p "Do you want to seed sample data? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Seeding sample data..."
    python scripts/seed_data.py
fi

# Run tests
read -p "Do you want to run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running tests..."
    pytest tests/ -v
fi

echo
echo "========================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "========================================="
echo
echo "Next steps:"
echo "1. Update .env with your configuration"
echo "2. Run 'source venv/bin/activate' to activate the environment"
echo "3. Run 'uvicorn app.main:app --reload' to start the server"
echo
