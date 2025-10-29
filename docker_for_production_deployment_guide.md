# Meticulous Plan for Production Docker Deployment

## Executive Summary

I'll create a comprehensive production deployment guide using Docker containers with Redis 7 and PostgreSQL 17. This plan involves a multi-container architecture with proper separation of concerns, optimized for production workloads.

## Deep Analysis of Current Architecture vs Production Requirements

### Current Development Setup
- SQLite database for development
- Redis connection with fallback to in-memory caching
- Single FastAPI application
- ChromaDB embedded in the application
- No dedicated reverse proxy

### Production Requirements
- PostgreSQL 17 for persistent data storage
- Redis 7 for distributed caching
- Nginx reverse proxy for load balancing and SSL termination
- Separate containers for frontend and backend
- Proper volume management for data persistence
- Environment-specific configuration
- Health checks and monitoring

## Detailed Implementation Plan

### Phase 1: Architecture Design

#### Container Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Nginx     │────│  Frontend   │    │   Backend   │    │    Redis    │    │ PostgreSQL  │
│  (Reverse   │    │   (React)   │    │  (FastAPI)  │    │   (Cache)   │    │ (Database)  │
│   Proxy)    │    │             │    │             │    │             │    │             │
│  Port:80/443│    │  Port:3000  │    │  Port:8000  │    │  Port:6379  │    │  Port:5432  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

#### Network Configuration
- Frontend network: For Nginx-Frontend communication
- Backend network: For Nginx-Backend-Redis-PostgreSQL communication
- Database network: For Backend-PostgreSQL communication

#### Volume Management
- PostgreSQL data volume: Persistent database storage
- Redis data volume: Persistent cache storage
- ChromaDB volume: Persistent vector database storage
- Nginx logs volume: Log persistence

### Phase 2: Configuration Files

#### 1. Dockerfile.prod (Main Application)
```dockerfile
# Multi-stage build for production
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production

COPY frontend/ ./
RUN npm run build

# Backend stage
FROM python:3.12-slim AS backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Python requirements
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Copy frontend build
COPY --from=frontend-builder /app/frontend/build ./static

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Dockerfile.nginx (Nginx Reverse Proxy)
```dockerfile
FROM nginx:1-alpine

# Remove default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom configuration
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/proxy.conf /etc/nginx/conf.d/proxy.conf

# Create non-root user
RUN addgroup -g 1001 -S nginx && \
    adduser -S nginx -u 1001

# Create directories for logs
RUN mkdir -p /var/log/nginx && \
    chown -R nginx:nginx /var/log/nginx

# Expose ports
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
```

#### 3. docker-compose.prod.yml
```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:17-alpine
    container_name: cs_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: cs_redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Backend Application
  backend:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: cs_backend
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - ENVIRONMENT=production
      - DEBUG=false
      - CORS_ORIGINS=${CORS_ORIGINS}
      - SECRET_KEY=${SECRET_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - chroma_data:/app/data
      - ./logs:/app/logs
    networks:
      - backend
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx Reverse Proxy
  nginx:
    build:
      context: .
      dockerfile: Dockerfile.nginx
    container_name: cs_nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    networks:
      - backend
    depends_on:
      - backend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 3s
      retries: 3

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  chroma_data:
    driver: local
  nginx_logs:
    driver: local

networks:
  backend:
    driver: bridge
```

#### 4. .env.docker
```env
# Database Configuration
POSTGRES_DB=customer_support
POSTGRES_USER=cs_user
POSTGRES_PASSWORD=CHANGE_ME_IN_PRODUCTION

# Redis Configuration
REDIS_PASSWORD=CHANGE_ME_IN_PRODUCTION

# Application Configuration
SECRET_KEY=CHANGE_ME_IN_PRODUCTION
OPENAI_API_KEY=CHANGE_ME_IN_PRODUCTION
CORS_ORIGINS=https://yourdomain.com

# SSL Configuration (if using HTTPS)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem
```

#### 5. .dockerignore
```
# Git
.git
.gitignore

# Documentation
README.md
docs/

# Dependencies
node_modules/
venv/
env/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Environment
.env
.env.local
.env.development
.env.test
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log

# Database
*.db
*.sqlite
*.sqlite3

# Docker
Dockerfile*
docker-compose*.yml
.dockerignore

# Frontend build artifacts
frontend/build/
frontend/dist/

# Backend data
data/
uploads/
temp/
```

#### 6. nginx/nginx.conf
```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging format
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Include proxy configuration
    include /etc/nginx/conf.d/proxy.conf;
}
```

#### 7. nginx/proxy.conf
```nginx
# Health check endpoint
server {
    listen 80;
    server_name _;
    
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}

# Main server configuration
server {
    listen 80;
    server_name yourdomain.com;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Frontend static files
    location / {
        root /var/www/html;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket endpoint
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific timeouts
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}

# HTTPS configuration (uncomment and configure for production)
# server {
#     listen 443 ssl http2;
#     server_name yourdomain.com;
#     
#     ssl_certificate /etc/nginx/ssl/cert.pem;
#     ssl_certificate_key /etc/nginx/ssl/key.pem;
#     
#     # SSL configuration
#     ssl_protocols TLSv1.2 TLSv1.3;
#     ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
#     ssl_prefer_server_ciphers off;
#     ssl_session_cache shared:SSL:10m;
#     ssl_session_timeout 10m;
#     
#     # Include the same location blocks as HTTP
#     include /etc/nginx/conf.d/proxy-locations.conf;
# }
```

#### 8. docker/postgres/init.sql
```sql
-- Create additional database user for the application
CREATE USER cs_app WITH PASSWORD 'CHANGE_ME_IN_PRODUCTION';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE customer_support TO cs_app;
GRANT USAGE ON SCHEMA public TO cs_app;
GRANT CREATE ON SCHEMA public TO cs_app;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### Phase 3: Backend Configuration Updates

#### 1. Update config.py for Production
```python
# Add these production-specific configurations

# PostgreSQL Configuration
database_url: str = Field(
    default="sqlite:///./data/customer_support.db",
    description="Database connection URL"
)

# Redis Configuration
redis_url: str = Field(
    default="redis://localhost:6379/0",
    description="Redis connection URL"
)

# Production-specific settings
if environment == Environment.PRODUCTION:
    # Override database URL for production
    database_url = os.environ.get("DATABASE_URL", database_url)
    
    # Override Redis URL for production
    redis_url = os.environ.get("REDIS_URL", redis_url)
    
    # Disable debug mode
    debug = False
    
    # Use production logging
    log_level = LogLevel.INFO
```

#### 2. Update database.py for PostgreSQL
```python
# Add PostgreSQL-specific configuration

def init_db() -> None:
    """Initialize database tables."""
    try:
        logger.info("Initializing database...")
        
        # Import all models to ensure they're registered with Base
        from .models import memory, session, message
        
        # Check if using PostgreSQL
        if "postgresql" in settings.database_url:
            logger.info("Using PostgreSQL database")
            # Create schema if it doesn't exist
            with engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS public"))
                conn.commit()
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}", exc_info=True)
        raise
```

### Phase 4: Deployment Scripts

#### 1. deploy.sh
```bash
#!/bin/bash

# Production deployment script
set -e

echo "Starting production deployment..."

# Check if .env.docker exists
if [ ! -f .env.docker ]; then
    echo "Error: .env.docker file not found. Please create it from the template."
    exit 1
fi

# Pull latest images
echo "Pulling latest images..."
docker-compose -f docker-compose.prod.yml pull

# Build and start services
echo "Building and starting services..."
docker-compose -f docker-compose.prod.yml up --build -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 30

# Check service health
echo "Checking service health..."
docker-compose -f docker-compose.prod.yml ps

echo "Deployment completed successfully!"
```

#### 2. backup.sh
```bash
#!/bin/bash

# Backup script for production data
set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

echo "Starting backup process..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
echo "Backing up PostgreSQL..."
docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U cs_user customer_support > $BACKUP_DIR/postgres_$DATE.sql

# Backup Redis
echo "Backing up Redis..."
docker-compose -f docker-compose.prod.yml exec -T redis redis-cli --rdb - > $BACKUP_DIR/redis_$DATE.rdb

# Backup ChromaDB
echo "Backing up ChromaDB..."
docker run --rm -v cs_chroma_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/chroma_$DATE.tar.gz -C /data .

echo "Backup completed successfully!"
```

## Validation Plan

### 1. Configuration Validation
- Verify all environment variables are properly set
- Check Docker Compose file syntax
- Validate Nginx configuration
- Test database connection strings

### 2. Container Health Checks
- Verify PostgreSQL container is healthy
- Verify Redis container is healthy
- Verify Backend container is healthy
- Verify Nginx container is healthy

### 3. Integration Testing
- Test API endpoints through Nginx proxy
- Test WebSocket connections
- Verify database operations
- Test cache functionality

### 4. Performance Testing
- Load test the application
- Monitor resource usage
- Check response times
- Verify scaling capabilities

## Implementation Checklist

- [ ] Create Dockerfile.prod for main application
- [ ] Create Dockerfile.nginx for reverse proxy
- [ ] Create docker-compose.prod.yml with all services
- [ ] Create .env.docker with production variables
- [ ] Create .dockerignore to optimize build context
- [ ] Create Nginx configuration files
- [ ] Create PostgreSQL initialization script
- [ ] Update backend configuration for production
- [ ] Create deployment scripts
- [ ] Create backup scripts
- [ ] Test all configurations
- [ ] Validate container health checks
- [ ] Perform integration testing
- [ ] Document deployment process

This comprehensive plan provides a production-ready Docker deployment with proper separation of concerns, data persistence, and monitoring capabilities. The implementation follows best practices for security, performance, and maintainability.

---

https://chat.z.ai/s/e0015b27-f0ee-4daf-9d27-cc8c6ed4e45c 

