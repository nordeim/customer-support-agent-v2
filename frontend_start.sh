# Install dependencies
cd frontend
npm install

# Set environment variables
cp .env.example .env
# Edit .env with your backend URL

# Development mode
npm run dev
# Open http://localhost:3000

# Build for production
npm run build

# Run with Docker
docker build -t customer-support-frontend .
docker run -p 80:80 customer-support-frontend
