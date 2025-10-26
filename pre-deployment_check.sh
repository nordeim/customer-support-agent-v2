# 1. Run all tests
./scripts/run_tests.sh

# 2. Check code quality
./scripts/lint.sh

# 3. Build Docker images
docker-compose build

# 4. Run locally
docker-compose up

# 5. Test endpoints
curl http://localhost:8000/health
curl http://localhost:3000
