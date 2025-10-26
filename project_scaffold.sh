# Commands to execute
mkdir customer-support-ai-agent
cd customer-support-ai-agent
git init

# Create directory structure
mkdir -p backend/app/{agents,tools,api/routes,models,services,utils}
mkdir -p frontend/src/{components,hooks,services,types}
mkdir -p monitoring scripts

# Initialize backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create requirements.txt with exact versions
cat > requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
sqlalchemy==2.0.35
alembic==1.13.3
chromadb==0.5.20
sentence-transformers==3.2.1
redis==5.2.0
markitdown==0.1.0
opentelemetry-api==1.29.0
opentelemetry-instrumentation-fastapi==0.49b0
prometheus-client==0.21.0
pydantic==2.9.2
pydantic-settings==2.6.1
python-dotenv==1.0.1
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
EOF

pip install -r requirements.txt
