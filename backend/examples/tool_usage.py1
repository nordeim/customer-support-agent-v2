"""
Examples of using the agent tools independently or together.
This file demonstrates how the tools work in practice.
"""
import asyncio
from pathlib import Path

from app.tools.rag_tool import RAGTool
from app.tools.memory_tool import MemoryTool
from app.tools.attachment_tool import AttachmentTool
from app.tools.escalation_tool import EscalationTool


async def example_rag_usage():
    """Example of using RAG tool for knowledge base search."""
    print("\n=== RAG Tool Example ===")
    
    rag_tool = RAGTool()
    
    # Add some documents to the knowledge base
    documents = [
        "Our customer support is available 24/7 via chat, email at support@example.com, or phone at 1-800-555-1234.",
        "The return policy allows customers to return items within 30 days of purchase for a full refund.",
        "Shipping typically takes 3-5 business days for standard delivery and 1-2 days for express shipping.",
        "To reset your password, click on 'Forgot Password' on the login page and follow the email instructions.",
        "Premium members get free shipping on all orders and priority customer support."
    ]
    
    print("Adding documents to knowledge base...")
    result = rag_tool.add_documents(documents)
    print(f"Added {result['documents_added']} documents")
    
    # Search the knowledge base
    queries = [
        "How can I contact support?",
        "What is the return policy?",
        "Password reset process"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        search_result = await rag_tool.search(query, k=2)
        
        for i, source in enumerate(search_result["sources"], 1):
            print(f"  Result {i} (relevance: {source['relevance_score']:.2f}):")
            print(f"    {source['content'][:100]}...")


async def example_memory_usage():
    """Example of using Memory tool for conversation context."""
    print("\n=== Memory Tool Example ===")
    
    memory_tool = MemoryTool()
    session_id = "example-session-001"
    
    # Store various types of memories
    print(f"Storing memories for session: {session_id}")
    
    await memory_tool.store_memory(
        session_id=session_id,
        content="Customer name: Alice Johnson",
        content_type="user_info",
        importance=0.9
    )
    
    await memory_tool.store_memory(
        session_id=session_id,
        content="Prefers email communication over phone",
        content_type="preference",
        importance=0.7
    )
    
    await memory_tool.store_memory(
        session_id=session_id,
        content="Previous issue with order #12345 - resolved",
        content_type="context",
        importance=0.6
    )
    
    # Retrieve memories
    print("\nRetrieving all memories:")
    memories = await memory_tool.retrieve_memories(session_id)
    for memory in memories:
        print(f"  [{memory['content_type']}] {memory['content']}")
    
    # Get session summary
    print("\nSession Summary:")
    summary = await memory_tool.summarize_session(session_id)
    print(f"  {summary}")


async def example_attachment_usage():
    """Example of using Attachment tool for document processing."""
    print("\n=== Attachment Tool Example ===")
    
    attachment_tool = AttachmentTool()
    
    # Create a sample file
    sample_file = Path("sample_document.txt")
    sample_file.write_text(
        """
        Customer Agreement Terms
        
        1. Service Level Agreement
        We guarantee 99.9% uptime for our services.
        
        2. Support Response Times
        - Critical issues: 1 hour
        - High priority: 4 hours  
        - Normal priority: 24 hours
        
        3. Data Security
        All customer data is encrypted at rest and in transit.
        We comply with GDPR and CCPA regulations.
        """
    )
    
    try:
        # Process the attachment
        print(f"Processing file: {sample_file}")
        result = await attachment_tool.process_attachment(
            str(sample_file),
            chunk_for_rag=True
        )
        
        if result["success"]:
            print(f"✓ Successfully processed {result['filename']}")
            print(f"  Word count: {result['word_count']}")
            print(f"  Preview: {result['preview'][:100]}...")
            
            if "chunks" in result:
                print(f"  Created {result['chunk_count']} chunks for RAG indexing")
        else:
            print(f"✗ Failed: {result['error']}")
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()


async def example_escalation_usage():
    """Example of using Escalation tool for detecting support escalation needs."""
    print("\n=== Escalation Tool Example ===")
    
    escalation_tool = EscalationTool()
    
    # Test various messages
    test_messages = [
        {
            "message": "Thank you for your help!",
            "expected": False
        },
        {
            "message": "This is urgent! My payment is stuck and I need this resolved immediately!",
            "expected": True
        },
        {
            "message": "I want to speak to a manager right now!",
            "expected": True
        },
        {
            "message": "This is the third time I'm explaining this issue...",
            "history": [
                {"role": "user", "content": "My order hasn't arrived"},
                {"role": "assistant", "content": "Let me check that for you"},
                {"role": "user", "content": "It's been 2 weeks already"},
            ],
            "expected": True
        }
    ]
    
    for test in test_messages:
        message = test["message"]
        print(f"\nAnalyzing: '{message[:50]}...'")
        
        result = await escalation_tool.should_escalate(
            message=message,
            message_history=test.get("history"),
            create_ticket=True
        )
        
        print(f"  Should escalate: {result['escalate']} (expected: {test['expected']})")
        print(f"  Confidence: {result['confidence']:.2f}")
        
        if result['reasons']:
            print(f"  Reasons: {', '.join(result['reasons'][:3])}")
        
        if result['escalate']:
            print(f"  Priority: {result.get('priority', 'normal')}")
            print(f"  Category: {result.get('category', 'general')}")


async def example_integrated_workflow():
    """Example of using multiple tools together in a workflow."""
    print("\n=== Integrated Workflow Example ===")
    print("Simulating a customer support interaction with multiple tools...\n")
    
    # Initialize tools
    rag_tool = RAGTool()
    memory_tool = MemoryTool()
    escalation_tool = EscalationTool()
    
    session_id = "integrated-example-001"
    
    # Simulate conversation flow
    messages = [
        "Hi, I need help with my recent order #98765",
        "The product arrived damaged and I want a refund",
        "This is really frustrating! I've been a customer for 5 years!"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"Step {i}: User says: '{message}'")
        
        # Check for escalation
        escalation_result = await escalation_tool.should_escalate(message)
        
        if escalation_result["escalate"]:
            print(f"  ⚠️ Escalation triggered (confidence: {escalation_result['confidence']:.2f})")
        
        # Store in memory
        await memory_tool.store_memory(
            session_id=session_id,
            content=message,
            content_type="context",
            importance=0.5 + (0.2 if escalation_result["escalate"] else 0)
        )
        
        # Search knowledge base
        search_result = await rag_tool.search(message, k=1)
        if search_result["sources"]:
            print(f"  📚 Found relevant info: {search_result['sources'][0]['content'][:80]}...")
        
        print()
    
    # Get final session summary
    print("Final Session Summary:")
    summary = await memory_tool.summarize_session(session_id)
    print(f"  {summary}")


async def main():
    """Run all examples."""
    await example_rag_usage()
    await example_memory_usage()
    await example_attachment_usage()
    await example_escalation_usage()
    await example_integrated_workflow()


if __name__ == "__main__":
    asyncio.run(main())
