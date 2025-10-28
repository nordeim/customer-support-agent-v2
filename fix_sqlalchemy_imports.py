#!/usr/bin/env python3
"""
Precision fix for missing SQLAlchemy type imports in backend/app/models/.

Scans model files and ensures all used SQLAlchemy column types
(e.g., Float, Boolean, etc.) are properly imported.

This script is idempotent and safe to run multiple times.
"""

import re
from pathlib import Path

# Project root and models directory
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "backend" / "app" / "models"

# SQLAlchemy column types we care about
SQLALCHEMY_TYPES = {"Float", "Boolean", "LargeBinary", "Enum"}

def get_used_types(content: str) -> set:
    """Extract SQLAlchemy column types used in Column(...) declarations."""
    used = set()
    # Match patterns like: Column(Float, ...)
    # or Column("name", Float, ...)
    pattern = r'Column\(\s*(?:"[^"]+"\s*,\s*)?([A-Z][a-zA-Z]*)'
    for match in re.finditer(pattern, content):
        type_name = match.group(1)
        if type_name in SQLALCHEMY_TYPES:
            used.add(type_name)
    return used

def fix_imports_in_file(filepath: Path) -> bool:
    """Fix missing imports in a single model file. Returns True if modified."""
    content = filepath.read_text()
    original_content = content

    # Find the sqlalchemy import line
    import_pattern = r'^(from sqlalchemy import .+)$'
    import_match = re.search(import_pattern, content, re.MULTILINE)
    
    if not import_match:
        print(f"  ⚠️  No sqlalchemy import found in {filepath.name}")
        return False

    import_line = import_match.group(1)
    current_imports = set()
    
    # Parse existing imports: handle both single-line and multi-line
    if import_line.endswith('\\'):
        # Multi-line import (not present in your files, but future-proofing)
        start_line = import_match.start(1)
        lines = content.splitlines()
        idx = lines.index(import_line.rstrip('\\'))
        full_import = lines[idx].rstrip('\\')
        while idx + 1 < len(lines) and lines[idx + 1].strip().endswith((')', ',')):
            idx += 1
            full_import += lines[idx].strip()
        # Extract imports between parentheses
        inside = re.search(r'import\s*\(([^)]+)\)', full_import, re.DOTALL)
        if inside:
            imports_text = inside.group(1)
            current_imports = {imp.strip() for imp in imports_text.replace('\\', '').split(',') if imp.strip()}
    else:
        # Single-line import
        inside = re.search(r'import\s+(.+)$', import_line)
        if inside:
            imports_text = inside.group(1)
            # Remove trailing comment if any
            imports_text = re.split(r'\s+#', imports_text)[0]
            current_imports = {imp.strip() for imp in imports_text.split(',') if imp.strip()}

    # Determine what types are used in this file
    used_types = get_used_types(content)
    missing_types = used_types - current_imports

    if not missing_types:
        return False  # No changes needed

    # Add missing types to the import line
    new_imports = sorted(current_imports | missing_types)
    new_import_line = f"from sqlalchemy import {', '.join(new_imports)}"
    
    # Replace the old import line
    content = re.sub(import_pattern, new_import_line, content, count=1, flags=re.MULTILINE)
    
    if content != original_content:
        filepath.write_text(content)
        return True
    
    return False

def main():
    print("🔍 Scanning backend/app/models/ for missing SQLAlchemy type imports...\n")
    
    changed_files = []
    
    for model_file in sorted(MODELS_DIR.glob("*.py")):
        if model_file.name == "__init__.py" or model_file.name == "schemas.py":
            continue
            
        print(f"  📄 Checking {model_file.name}...")
        if fix_imports_in_file(model_file):
            print(f"  ✅ Fixed missing imports in {model_file.name}")
            changed_files.append(model_file)
        else:
            print(f"  ⚪ No changes needed for {model_file.name}")
    
    print(f"\n✅ Scan complete. {len(changed_files)} file(s) updated.")
    if changed_files:
        print("\nChanged files:")
        for f in changed_files:
            print(f"  - {f.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
