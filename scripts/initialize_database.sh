# Backup database first!
cp backend/data/customer_support.db backend/data/customer_support.db.backup

# Run migration
alembic upgrade head

# Verify migration
python -c "
from backend.app.database import engine
from sqlalchemy import inspect
inspector = inspect(engine)
columns = [c['name'] for c in inspector.get_columns('memories')]
print('Columns:', columns)
assert 'content_hash' in columns, 'Migration failed!'
print('✓ Migration successful')
"
