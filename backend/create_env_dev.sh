cd backend
cat > .env << 'EOF'
# Minimal working configuration
APP_NAME=Customer Support AI Agent
ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000","http://localhost:3001"]

# Database
DATABASE_URL=sqlite:///./data/customer_support.db

# Redis (disable if not running)
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=false

# AI Provider (use local for testing)
AI_PROVIDER=local
DEV_MOCK_AI=true
EOF
