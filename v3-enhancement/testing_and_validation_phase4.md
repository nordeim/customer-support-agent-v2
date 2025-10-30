# 1. Install new dependencies
pip install -r requirements.txt

# 2. Run Phase 4 session store tests (without Redis)
./scripts/run_tests.sh tests/test_session_store.py -m "not integration" -v

# 3. Run with Redis (requires Redis running on localhost:6379)
docker run -d -p 6379:6379 redis:7-alpine
./scripts/run_tests.sh tests/test_session_store.py -v

# 4. Run all tests
./scripts/run_tests.sh

# 5. Test in-memory mode (default)
export USE_SHARED_CONTEXT=false
python -m app.main

# 6. Test Redis mode (requires Redis)
export USE_SHARED_CONTEXT=true
export REDIS_URL=redis://localhost:6379/0
python -m app.main
