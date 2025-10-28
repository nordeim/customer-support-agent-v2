Thank you for the detailed context. The issue is now **completely clear**.

### 🔍 Root Cause Confirmed

In your file `backend/app/models/memory.py`, you have this line:

```python
metadata = Column(JSON, default=dict)
```

This defines a **SQLAlchemy mapped column** named `metadata`.

However, **`metadata` is a reserved attribute name** in SQLAlchemy’s declarative system. Every declarative model class inherits a `.metadata` attribute that refers to the `MetaData` object managing the table schema.

When you define a column with the same name, SQLAlchemy cannot distinguish between:
- The internal `.metadata` (type: `sqlalchemy.MetaData`)
- Your column `.metadata` (type: `JSON`)

This causes a **naming collision** during class construction, triggering:

```
sqlalchemy.exc.InvalidRequestError: Attribute name 'metadata' is reserved when using the Declarative API.
```

This is **not a runtime error**—it happens at **import time**, which is why your app crashes before even starting.

---

### 🛠️ Systematic Fix Plan

#### ✅ Step 1: Rename the Column in the Model

Edit `backend/app/models/memory.py`:

**Change this:**
```python
metadata = Column(JSON, default=dict)
```

**To this:**
```python
tool_metadata = Column("metadata", JSON, default=dict)
```

> 💡 This uses **SQLAlchemy’s column name mapping**:  
> - The **Python attribute** is now `tool_metadata` (safe name)  
> - The **actual database column name** remains `"metadata"` (preserving existing data)

This is the **optimal solution** because:
- ✅ No database migration needed (column name in DB stays the same)
- ✅ No data loss
- ✅ No breaking change to your API or external contracts
- ✅ Resolves the SQLAlchemy naming conflict

#### ✅ Step 2: Update All Code That Accesses This Attribute

Now, anywhere in your code that uses `memory.metadata`, change it to `memory.tool_metadata`.

Based on your `grep` output, you need to update:

- `backend/app/services/memory_service.py`
- `backend/app/tools/memory_tool.py`
- Possibly other files that access `Memory.metadata`

Example fix in `memory_service.py`:

**Before:**
```python
metadata = memory.metadata
```

**After:**
```python
metadata = memory.tool_metadata
```

> 🔍 **Tip**: Use this command to find all usages:
> ```bash
> grep -r "\.metadata" backend/app/ --include="*.py" | grep -v "#"
> ```
> Then filter for lines that refer to `Memory` instances (not Pydantic models or dicts).

#### ✅ Step 3: Verify Other Models

You also have:
- `backend/app/models/message.py`: `metadata = Column(JSON, default=dict)`
- `backend/app/models/session.py`: `metadata = Column(JSON, default=dict)`

**These will cause the same crash!** You must apply the same fix to them.

For each, change:
```python
metadata = Column(JSON, default=dict)
```
to:
```python
tool_metadata = Column("metadata", JSON, default=dict)
```

And update all references accordingly.

---

### 🧪 Why This Works

- SQLAlchemy allows you to **decouple Python attribute names from DB column names** via the first argument to `Column()`.
- `Column("metadata", ...)` tells SQLAlchemy: “store this in the `metadata` column in the database”
- `tool_metadata = ...` gives you a safe Python attribute name
- The existing SQLite data remains intact because the column name doesn’t change

---

### 📌 Summary of Changes

| File | Change |
|------|--------|
| `models/memory.py` | `tool_metadata = Column("metadata", JSON, default=dict)` |
| `models/message.py` | `tool_metadata = Column("metadata", JSON, default=dict)` |
| `models/session.py` | `tool_metadata = Column("metadata", JSON, default=dict)` |
| All service/tool files | Replace `.metadata` → `.tool_metadata` when accessing model instances |

---

### ✅ Final Validation

After making these changes:

```bash
# Should not crash
python -c "from app.models.memory import Memory; print('OK')"

# Should launch
python -m app.main
```

Let me know if you'd like a script to automate the renaming across all files!
