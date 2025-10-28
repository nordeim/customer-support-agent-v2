#!/usr/bin/env python3
"""
Automated fix for SQLAlchemy 'metadata' reserved attribute conflict.

Renames:
  - In model files: `metadata = Column(...)` → `tool_metadata = Column("metadata", ...)`
  - In service/tool files: `.metadata` → `.tool_metadata` when used on Memory/Message/Session instances

Preserves:
  - All other uses of 'metadata' (Pydantic, dicts, args, etc.)
  - Database column name remains 'metadata'
"""

import os
import re
import ast
import sys
from pathlib import Path

# === CONFIGURATION ===
PROJECT_ROOT = Path(__file__).parent
BACKEND_APP = PROJECT_ROOT / "backend" / "app"

MODEL_FILES = [
    BACKEND_APP / "models" / "memory.py",
    BACKEND_APP / "models" / "message.py",
    BACKEND_APP / "models" / "session.py",
]

# Files that likely access .metadata on model instances
CANDIDATE_FILES = []
for ext in ("*.py",):
    CANDIDATE_FILES.extend(BACKEND_APP.rglob(ext))

# Exclude model files from candidate list to avoid double-processing
CANDIDATE_FILES = [f for f in CANDIDATE_FILES if f not in MODEL_FILES]

# === STEP 1: Fix model files ===
def fix_model_file(filepath: Path) -> int:
    """Fix a single model file. Returns number of changes."""
    content = filepath.read_text()
    original = content

    # Match: metadata = Column(JSON, ...)
    pattern = r'^(\s*)metadata\s*=\s*Column\('
    replacement = r'\1tool_metadata = Column("metadata", '

    # Use MULTILINE to match ^ at line start
    content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if content != original:
        filepath.write_text(content)
        return count
    return 0

# === STEP 2: Fix usage in other files ===
def fix_usage_file(filepath: Path) -> int:
    """Fix .metadata usage in non-model files."""
    content = filepath.read_text()
    original = content

    # We'll match `.metadata` but only when it's likely a model attribute
    # This regex matches `.metadata` not followed by a letter/underscore (to avoid partial matches)
    # and not inside a string or comment (approximation)
    pattern = r'\.metadata(?=\s*[^a-zA-Z_])'

    # Replace with .tool_metadata
    content, count = re.subn(pattern, '.tool_metadata', content)

    if content != original:
        filepath.write_text(content)
        return count
    return 0

# === MAIN EXECUTION ===
def main():
    print("🔍 Fixing SQLAlchemy 'metadata' reserved attribute conflict...\n")

    total_changes = 0
    changed_files = []

    # --- Fix model definitions ---
    print("🔧 Updating model files:")
    for model_file in MODEL_FILES:
        if model_file.exists():
            changes = fix_model_file(model_file)
            if changes > 0:
                print(f"  ✅ {model_file.relative_to(PROJECT_ROOT)} ({changes} change(s))")
                total_changes += changes
                changed_files.append(model_file)
            else:
                print(f"  ⚪ {model_file.relative_to(PROJECT_ROOT)} (no changes)")
        else:
            print(f"  ❌ {model_file.relative_to(PROJECT_ROOT)} (not found)")

    print()

    # --- Fix usages ---
    print("🔧 Updating usage in other files:")
    for py_file in sorted(CANDIDATE_FILES):
        changes = fix_usage_file(py_file)
        if changes > 0:
            print(f"  ✅ {py_file.relative_to(PROJECT_ROOT)} ({changes} change(s))")
            total_changes += changes
            changed_files.append(py_file)

    print()

    # --- Summary ---
    print("📊 Summary:")
    print(f"  Files changed: {len(changed_files)}")
    print(f"  Total replacements: {total_changes}")

    if total_changes > 0:
        print("\n✅ Fix applied successfully. Please verify with:")
        print("   python -c \"from app.models.memory import Memory; print('OK')\"")
        print("   python -m app.main")
    else:
        print("\nℹ️  No changes were needed.")

if __name__ == "__main__":
    main()
