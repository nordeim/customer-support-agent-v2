# Run all tests
cd backend
pytest tests/test_tools.py -v

# Run examples
python examples/tool_usage.py

# Test individual tools
python -c "from app.tools.rag_tool import RAGTool; tool = RAGTool(); print('RAG Tool initialized')"
