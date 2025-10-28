Thank you for the detailed traceback and for trusting me to dig deeper.

This new error is **not** about a missing import like `Integer`—it’s more subtle and structural:

```
sqlalchemy.exc.InvalidRequestError: Attribute name 'metadata' is reserved when using the Declarative API.
```

This is a **critical SQLAlchemy-specific error** that occurs when a model class defines a column or attribute named `metadata`, which **collides with SQLAlchemy’s internal `metadata` attribute** used to track table schema information.

---

### 🔍 Step-by-Step Root Cause Analysis

#### 1. **Where is this happening?**
- File: `backend/app/models/memory.py`
- Class: `Memory(Base)`
- Error raised during class creation (i.e., at import time)
- SQLAlchemy’s declarative base (`Base`) uses an internal attribute called `.metadata` (an instance of `MetaData`) to manage table definitions.
- If your model defines a **column or class attribute** named `metadata`, SQLAlchemy **cannot distinguish** it from its own internal one → **raises `InvalidRequestError`**.

> ✅ Confirmed by SQLAlchemy docs:  
> > _“The name `metadata` is reserved for the MetaData collection and cannot be used as a mapped attribute.”_  
> — [SQLAlchemy Declarative Docs](https://docs.sqlalchemy.org/en/20/orm/declarative_config.html#reserved-attribute-names)

#### 2. **What’s likely in `memory.py`?**
Even though you haven’t shared the full content of `memory.py`, the error + your project context strongly suggests something like:

```python
class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    content = Column(Text)
    metadata = Column(JSON)  # ❌ THIS IS THE PROBLEM
    timestamp = Column(DateTime, default=datetime.utcnow)
```

Here, `metadata = Column(JSON)` is **illegal** under SQLAlchemy’s declarative system.

---

### 🛠️ How to Fix It

You **must rename the column** to something else—e.g., `meta`, `extra_data`, `context_metadata`, or `tool_metadata`.

#### ✅ Recommended Fix:

**Step 1: Edit `backend/app/models/memory.py`**

Change:
```python
metadata = Column(JSON)
```

To:
```python
tool_metadata = Column(JSON, nullable=True)  # or 'meta_data', 'context_data', etc.
```

> 💡 Choose a name that reflects its purpose. Since this likely stores metadata from RAG sources or tool calls (e.g., file name, page number, relevance score), `source_metadata` or `tool_metadata` is semantically accurate.

**Step 2: Update all code that references `.metadata` on `Memory` objects**

Search your codebase for:
```bash
grep -r "\.metadata" backend/app/
```

You’ll likely find usage in:
- `memory_service.py`
- `rag_tool.py`
- `chat_agent.py`
- Possibly in API response serializers

Update each reference:
```python
# Before
mem.metadata

# After
mem.tool_metadata
```

**Step 3: Update Pydantic schemas (if any)**

If you have a Pydantic model like:

```python
class MemoryResponse(BaseModel):
    metadata: dict | None = None
```

You should also rename the field there for consistency (or use `Field(alias="metadata")` if you want to keep the external API field name as `"metadata"` while using `tool_metadata` internally).

Example:
```python
from pydantic import BaseModel, Field

class MemoryResponse(BaseModel):
    tool_metadata: dict | None = Field(alias="metadata")

    class Config:
        populate_by_name = True
```

But **only do this if you need to preserve the API contract**. Otherwise, just rename it everywhere.

**Step 4: Handle database migration**

Since you’re changing a column name:
- If you’re using **SQLite in dev** and don’t care about preserving data → just delete `customer_support.db` and let Alembic or app startup recreate it.
- If you’re using **PostgreSQL or need to preserve data**, you **must create an Alembic migration**:

```bash
cd backend
alembic revision -m "rename metadata column to tool_metadata in memories table"
```

Then edit the migration file:

```python
def upgrade() -> None:
    op.alter_column('memories', 'metadata', new_column_name='tool_metadata')

def downgrade() -> None:
    op.alter_column('memories', 'tool_metadata', new_column_name='metadata')
```

> ⚠️ But note: SQLite doesn’t support `ALTER COLUMN ... RENAME` directly in older versions. Alembic will simulate it via table recreation.

---

### 🔎 Why Didn’t This Show Up Earlier?

- The error only occurs **at import time** when SQLAlchemy processes the model class.
- If the model was never imported (e.g., during unit tests that mock DB), it wouldn’t surface.
- It’s a **static structural error**, not a runtime logic bug.

---

### ✅ Validation Plan

After the fix:

1. **Verify import works**:
   ```bash
   python -c "from app.models.memory import Memory; print('OK')"
   ```

2. **Launch app**:
   ```bash
   python -m app.main
   ```

3. **Test memory creation** via API or agent flow.

---

### 📌 Summary

| Issue | Cause | Fix |
|------|------|-----|
| `Attribute name 'metadata' is reserved` | Column named `metadata` in SQLAlchemy model | Rename column to `tool_metadata` (or similar) and update all references |

---

### 🔜 Next Steps

Please confirm:
- Can you share the full content of `backend/app/models/memory.py` so I can verify the exact column definition?
- Are you using SQLite in development? (If yes, we can suggest a quick reset strategy.)

I’m ready to help you craft the exact code changes and migration if needed.
