# Make sure the server is running first
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run the test
cd backend
python test_backend.py

# Or run with custom server URL
API_URL=http://localhost:8000 python test_backend.py

# Quick test
# For a quick smoke test with minimal dependencies
python quick_test.py
