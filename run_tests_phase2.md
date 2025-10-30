# 1. Run Phase 2 tests
./scripts/run_tests.sh tests/test_registry.py -v

# 2. Run all tests
./scripts/run_tests.sh

# 3. Test legacy mode (default)
export AGENT_TOOL_REGISTRY_MODE=legacy
python -m app.main

# 4. Test registry mode
export AGENT_TOOL_REGISTRY_MODE=registry
python -m app.main
