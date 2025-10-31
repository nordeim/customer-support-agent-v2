# Test that all imports work
python -c "from backend.app.schemas.tool_requests import MemoryStoreRequest; print('✓ Schemas OK')"
python -c "from backend.app.tools.memory_tool import MemoryTool; print('✓ Memory tool OK')"
python -c "from backend.app.config.tool_settings import tool_settings; print('✓ Settings OK')"
python -c "from backend.app.models.memory import Memory; print('✓ Model OK')"
