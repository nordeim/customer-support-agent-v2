# Backend testing
cd backend
pytest tests/ -v --cov=app --cov-report=html

# Frontend testing  
cd frontend
npm test -- --coverage
