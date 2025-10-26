#!/usr/bin/env python3
"""
Seed sample data for development and testing.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import logging
from datetime import datetime, timedelta
import random
import uuid

from app.database import init_db, get_db
from app.models.session import Session
from app.models.message import Message
from app.models.memory import Memory
from app.tools.rag_tool import RAGTool
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample knowledge base documents
SAMPLE_DOCUMENTS = [
    # Policies
    "Return Policy: Customers can return items within 30 days of purchase for a full refund. Items must be in original condition with tags attached.",
    "Shipping Policy: Standard shipping takes 3-5 business days and costs $9.99. Free shipping on orders over $50.",
    "Privacy Policy: We protect your personal data and never share it with third parties without consent.",
    
    # FAQs
    "How to reset password: Click 'Forgot Password' on the login page, enter your email, and follow the instructions sent to your inbox.",
    "Order tracking: Use your order number on our tracking page or contact support@example.com for assistance.",
    "Payment methods: We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay.",
    
    # Product Information
    "Premium membership benefits: Free shipping on all orders, priority customer support, exclusive discounts, early access to sales.",
    "Account verification: Required for security. Please provide a valid email address and phone number.",
    "Technical support: Available 24/7 via chat. Phone support available Monday-Friday 9AM-6PM EST.",
    
    # Troubleshooting
    "Login issues: Clear your browser cache, try a different browser, or reset your password if you've forgotten it.",
    "Payment failures: Check card details, ensure sufficient funds, try a different payment method, or contact your bank.",
    "Delivery delays: Check tracking information, verify shipping address, contact carrier directly for updates.",
]

# Sample conversation scenarios
SAMPLE_CONVERSATIONS = [
    {
        "user_messages": [
            "Hi, I need help with my order",
            "Order number is #12345",
            "It hasn't arrived yet and it's been 7 days"
        ],
        "assistant_messages": [
            "Hello! I'd be happy to help you with your order.",
            "Thank you for providing your order number #12345. Let me check the status for you.",
            "I can see your order was shipped 7 days ago. According to our shipping policy, standard delivery takes 3-5 business days. Let me check with the carrier for any delays."
        ]
    },
    {
        "user_messages": [
            "How do I return an item?",
            "I bought it last week",
            "Thanks!"
        ],
        "assistant_messages": [
            "I can help you with the return process. According to our return policy, you can return items within 30 days of purchase.",
            "Since you purchased it last week, you're well within the return window. Please ensure the item is in original condition with tags attached.",
            "You're welcome! To start your return, please visit our returns page or reply with your order number for further assistance."
        ]
    }
]

async def seed_knowledge_base():
    """Seed the RAG knowledge base with sample documents."""
    logger.info("Seeding knowledge base...")
    
    try:
        rag_tool = RAGTool()
        
        # Add sample documents
        result = rag_tool.add_documents(
            documents=SAMPLE_DOCUMENTS,
            metadatas=[
                {"category": "policy", "type": "return"} if "return" in doc.lower()
                else {"category": "policy", "type": "shipping"} if "shipping" in doc.lower()
                else {"category": "faq", "type": "general"}
                for doc in SAMPLE_DOCUMENTS
            ]
        )
        
        if result["success"]:
            logger.info(f"✓ Added {result['documents_added']} documents to knowledge base")
        else:
            logger.error(f"Failed to add documents: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error seeding knowledge base: {e}")

def seed_database():
    """Seed the database with sample sessions and messages."""
    logger.info("Seeding database...")
    
    db = next(get_db())
    
    try:
        # Create sample sessions
        sessions_created = 0
        messages_created = 0
        memories_created = 0
        
        for i, convo in enumerate(SAMPLE_CONVERSATIONS):
            # Create session
            session_id = f"sample_session_{uuid.uuid4().hex[:8]}"
            session = Session(
                id=session_id,
                user_id=f"sample_user_{i+1}",
                status="ended",
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 7)),
                last_activity=datetime.utcnow() - timedelta(hours=random.randint(1, 24))
            )
            db.add(session)
            sessions_created += 1
            
            # Create messages
            for j, (user_msg, assistant_msg) in enumerate(
                zip(convo["user_messages"], convo["assistant_messages"])
            ):
                # User message
                user_message = Message(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="user",
                    content=user_msg,
                    created_at=datetime.utcnow() - timedelta(hours=random.randint(2, 48))
                )
                db.add(user_message)
                messages_created += 1
                
                # Assistant message
                assistant_message = Message(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content=assistant_msg,
                    created_at=user_message.created_at + timedelta(seconds=random.randint(1, 5)),
                    processing_time=random.uniform(0.5, 2.0)
                )
                db.add(assistant_message)
                messages_created += 1
            
            # Create sample memories
            memory = Memory(
                id=str(uuid.uuid4()),
                session_id=session_id,
                content_type="context",
                content=f"User inquired about: {convo['user_messages'][0]}",
                importance=0.7,
                created_at=datetime.utcnow() - timedelta(hours=random.randint(1, 24))
            )
            db.add(memory)
            memories_created += 1
        
        # Commit all changes
        db.commit()
        
        logger.info(f"✓ Created {sessions_created} sessions")
        logger.info(f"✓ Created {messages_created} messages")
        logger.info(f"✓ Created {memories_created} memories")
        
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

async def main():
    """Main seeding function."""
    logger.info("="*50)
    logger.info("Starting data seeding...")
    logger.info("="*50)
    
    # Initialize database
    init_db()
    
    # Seed knowledge base
    await seed_knowledge_base()
    
    # Seed database
    seed_database()
    
    logger.info("="*50)
    logger.info("Data seeding completed!")
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(main())
