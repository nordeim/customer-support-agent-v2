# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Run the application
python -m app.main

# 4. Test endpoints
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}'

# Send message
curl -X POST http://localhost:8000/api/chat/sessions/{session_id}/messages \
  -F "message=Hello, I need help with my order" \
  -F "attachments=@document.pdf"

# 5. Access documentation
# Open http://localhost:8000/docs
