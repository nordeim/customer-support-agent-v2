The error you're encountering:

```
NameError: name 'Integer' is not defined
```

is **unrelated to PyTorch or your GPU**, and instead points to a **missing SQLAlchemy import** in your model file.

---

### 🔍 Root Cause

In your file:

```
backend/app/models/memory.py
```

You are using:

```python
access_count = Column(Integer, default=0)
```

But `Integer` is not imported. In SQLAlchemy, column types like `Integer`, `String`, `DateTime`, etc., must be explicitly imported from `sqlalchemy`.

---

### ✅ Fix

Open `backend/app/models/memory.py` and **add the missing import**.

#### Before (likely):
```python
from sqlalchemy import Column, ForeignKey, DateTime, Text
from ..database import Base
```

#### After (fixed):
```python
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from ..database import Base
```

> 💡 Even if `String` isn’t used in this file, it’s common to include it for consistency. The critical missing piece is **`Integer`**.

---

### 🔧 Additional Recommendations

1. **Check other model files** (`session.py`, `message.py`) for similar missing imports.
   - Common missing types: `Integer`, `String`, `Boolean`, `Float`, `JSON` (if used).

2. **Enable static type checking** (you already have `mypy` in `requirements.txt`):
   ```bash
   mypy backend/app/models/
   ```
   This would have caught the undefined name at dev time.

3. **Add a linter check** (e.g., `flake8` or `ruff`) which can detect undefined names:
   ```bash
   flake8 backend/app/models/memory.py
   # Should report: F821 undefined name 'Integer'
   ```

---

### 🧪 Verify the Fix

After adding the import:

```bash
python -c "from app.models.memory import Memory; print('Success')"
```

Then try launching again:

```bash
python -m app.main
```

The segmentation fault should now be gone (thanks to CPU-only PyTorch), and this `NameError` will be resolved.

Let me know if you'd like a script to auto-audit all model files for missing SQLAlchemy imports!
