# 1. Install new dependencies
pip install -r requirements.txt

# 2. Run Phase 3 integration tests
./scripts/run_tests.sh tests/test_agent_tool_integration.py -v

# 3. Run all tests
./scripts/run_tests.sh

# 4. Start application
python -m app.main
