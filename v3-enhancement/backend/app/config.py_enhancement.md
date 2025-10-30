config.py → 🟡 Optional enhancement:

Current version works perfectly for Phase 1
If you want Phase 2 preparation, add one line:

```
agent_tool_registry_mode: str = Field(default="legacy", description="Tool mode")
```
