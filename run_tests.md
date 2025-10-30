# 1. Make test script executable
chmod +x scripts/run_tests.sh

# 2. Run Phase 0 baseline tests
./scripts/run_tests.sh -m unit

# 3. Run Phase 1 async contract tests
./scripts/run_tests.sh

# 4. Run with coverage
./scripts/run_tests.sh --fail-under 80

# 5. Start application
python -m app.main
