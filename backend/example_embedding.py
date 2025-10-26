# Using Embedding Service
from app.services import embedding_service

# Generate embeddings
embedding = await embedding_service.generate_embedding("Hello world")
embeddings = await embedding_service.generate_embeddings(["Text 1", "Text 2"])

# Using Memory Service
from app.services import memory_service

# Store conversation
await memory_service.add_conversation_turn(
    session_id="sess_123",
    user_message="What's my order status?",
    assistant_message="Your order #12345 is in transit",
    confidence=0.95
)

# Get session context
context = await memory_service.get_session_context("sess_123")
