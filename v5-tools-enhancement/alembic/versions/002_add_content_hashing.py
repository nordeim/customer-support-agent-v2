"""
Add content hashing to memories table

Revision ID: 002_add_content_hashing
Revises: 001_initial_schema
Create Date: 2024-01-15 10:00:00.000000

Migration: Add content hashing and TTL support to memories table

Changes:
- Add content_hash column for duplicate detection
- Add semantic_hash column for similarity detection
- Add expires_at column for TTL support
- Add deleted_at column for soft delete
- Update unique constraint to use content_hash
- Add optimized indexes
- Backfill content_hash for existing data
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
import hashlib
import logging

# Revision identifiers
revision = '002_add_content_hashing'
down_revision = '001_initial_schema'  # Update this to match your previous migration
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)


def normalize_content(content: str) -> str:
    """Normalize content for hashing (matches Memory model)."""
    normalized = content.lower()
    normalized = ' '.join(normalized.split())
    normalized = normalized.rstrip('.,!?;:')
    return normalized


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of normalized content."""
    normalized = normalize_content(content)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def upgrade():
    """
    Apply migration: Add content hashing and TTL support.
    
    Steps:
    1. Add new columns (nullable initially)
    2. Backfill content_hash for existing data
    3. Make content_hash non-nullable
    4. Drop old unique constraint
    5. Add new unique constraint on content_hash
    6. Add indexes
    """
    logger.info("Starting migration: Add content hashing to memories")
    
    # Get database connection
    connection = op.get_bind()
    is_postgresql = 'postgresql' in str(connection.engine.url)
    
    # ===========================
    # Step 1: Add new columns
    # ===========================
    
    logger.info("Adding new columns...")
    
    # Add content_hash (nullable initially for backfill)
    op.add_column(
        'memories',
        sa.Column('content_hash', sa.String(64), nullable=True)
    )
    
    # Add semantic_hash (optional, nullable)
    op.add_column(
        'memories',
        sa.Column('semantic_hash', sa.String(64), nullable=True)
    )
    
    # Add expires_at (optional TTL)
    op.add_column(
        'memories',
        sa.Column(
            'expires_at',
            sa.DateTime(timezone=True),
            nullable=True
        )
    )
    
    # Add deleted_at (soft delete)
    op.add_column(
        'memories',
        sa.Column(
            'deleted_at',
            sa.DateTime(timezone=True),
            nullable=True
        )
    )
    
    logger.info("✓ New columns added")
    
    # ===========================
    # Step 2: Backfill content_hash
    # ===========================
    
    logger.info("Backfilling content_hash for existing data...")
    
    if is_postgresql:
        # PostgreSQL: Use native hashing (faster)
        logger.info("Using PostgreSQL native hashing")
        
        connection.execute(text("""
            UPDATE memories
            SET content_hash = encode(
                digest(
                    lower(trim(trailing '.,!?;:' from regexp_replace(content, E'\\s+', ' ', 'g'))),
                    'sha256'
                ),
                'hex'
            )
            WHERE content_hash IS NULL
        """))
        
        logger.info("✓ Backfill completed using PostgreSQL digest")
    
    else:
        # SQLite or other: Use Python hashing
        logger.info("Using Python hashing for backfill")
        
        # Fetch all memories without content_hash
        result = connection.execute(
            text("SELECT id, content FROM memories WHERE content_hash IS NULL")
        )
        
        rows = result.fetchall()
        total_rows = len(rows)
        
        logger.info(f"Backfilling {total_rows} rows...")
        
        # Update in batches
        batch_size = 100
        for i in range(0, total_rows, batch_size):
            batch = rows[i:i + batch_size]
            
            for row in batch:
                memory_id = row[0]
                content = row[1]
                
                # Compute hash
                content_hash = compute_content_hash(content)
                
                # Update row
                connection.execute(
                    text("UPDATE memories SET content_hash = :hash WHERE id = :id"),
                    {"hash": content_hash, "id": memory_id}
                )
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Processed {i + batch_size}/{total_rows} rows...")
        
        logger.info("✓ Backfill completed using Python")
    
    # Commit backfill
    connection.execute(text("COMMIT"))
    
    # ===========================
    # Step 3: Make content_hash non-nullable
    # ===========================
    
    logger.info("Making content_hash non-nullable...")
    
    op.alter_column(
        'memories',
        'content_hash',
        nullable=False
    )
    
    logger.info("✓ content_hash is now non-nullable")
    
    # ===========================
    # Step 4: Update unique constraint
    # ===========================
    
    logger.info("Updating unique constraint...")
    
    # Drop old unique constraint (on full content)
    try:
        op.drop_constraint(
            'uq_memory_session_content',
            'memories',
            type_='unique'
        )
        logger.info("✓ Dropped old unique constraint")
    except Exception as e:
        logger.warning(f"Could not drop old constraint (may not exist): {e}")
    
    # Add new unique constraint (on content_hash)
    op.create_unique_constraint(
        'uq_memory_session_content_hash',
        'memories',
        ['session_id', 'content_type', 'content_hash']
    )
    
    logger.info("✓ Created new unique constraint on content_hash")
    
    # ===========================
    # Step 5: Add indexes
    # ===========================
    
    logger.info("Creating indexes...")
    
    # Index on content_hash
    op.create_index(
        'ix_memory_content_hash',
        'memories',
        ['content_hash']
    )
    
    # Index on semantic_hash
    op.create_index(
        'ix_memory_semantic_hash',
        'memories',
        ['semantic_hash']
    )
    
    # Index on expires_at
    op.create_index(
        'ix_memory_expires_at',
        'memories',
        ['expires_at']
    )
    
    # Index on deleted_at
    op.create_index(
        'ix_memory_deleted_at',
        'memories',
        ['deleted_at']
    )
    
    # Composite index for active memories (PostgreSQL partial index)
    if is_postgresql:
        # Partial index: only active (non-deleted) memories
        op.execute(text("""
            CREATE INDEX ix_memory_active_expires
            ON memories (expires_at)
            WHERE deleted_at IS NULL AND expires_at IS NOT NULL
        """))
        
        logger.info("✓ Created partial index for active memories")
    
    logger.info("✓ All indexes created")
    
    # ===========================
    # Step 6: Add check constraints
    # ===========================
    
    logger.info("Adding check constraints...")
    
    # Importance range check
    op.create_check_constraint(
        'ck_memory_importance_range',
        'memories',
        'importance >= 0.0 AND importance <= 1.0'
    )
    
    # Access count positive check
    op.create_check_constraint(
        'ck_memory_access_count_positive',
        'memories',
        'access_count >= 0'
    )
    
    logger.info("✓ Check constraints added")
    
    logger.info("✅ Migration completed successfully")


def downgrade():
    """
    Reverse migration: Remove content hashing and TTL support.
    
    WARNING: This will lose content_hash, semantic_hash, expires_at, and deleted_at data.
    """
    logger.info("Starting downgrade: Remove content hashing from memories")
    
    connection = op.get_bind()
    is_postgresql = 'postgresql' in str(connection.engine.url)
    
    # Drop check constraints
    logger.info("Dropping check constraints...")
    try:
        op.drop_constraint('ck_memory_access_count_positive', 'memories', type_='check')
        op.drop_constraint('ck_memory_importance_range', 'memories', type_='check')
    except Exception as e:
        logger.warning(f"Could not drop check constraints: {e}")
    
    # Drop indexes
    logger.info("Dropping indexes...")
    
    if is_postgresql:
        try:
            op.execute(text("DROP INDEX IF EXISTS ix_memory_active_expires"))
        except Exception as e:
            logger.warning(f"Could not drop partial index: {e}")
    
    try:
        op.drop_index('ix_memory_deleted_at', 'memories')
        op.drop_index('ix_memory_expires_at', 'memories')
        op.drop_index('ix_memory_semantic_hash', 'memories')
        op.drop_index('ix_memory_content_hash', 'memories')
    except Exception as e:
        logger.warning(f"Could not drop some indexes: {e}")
    
    # Drop new unique constraint
    logger.info("Dropping new unique constraint...")
    try:
        op.drop_constraint('uq_memory_session_content_hash', 'memories', type_='unique')
    except Exception as e:
        logger.warning(f"Could not drop new constraint: {e}")
    
    # Recreate old unique constraint
    logger.info("Recreating old unique constraint...")
    try:
        op.create_unique_constraint(
            'uq_memory_session_content',
            'memories',
            ['session_id', 'content_type', 'content']
        )
    except Exception as e:
        logger.warning(f"Could not recreate old constraint: {e}")
    
    # Drop new columns
    logger.info("Dropping new columns...")
    op.drop_column('memories', 'deleted_at')
    op.drop_column('memories', 'expires_at')
    op.drop_column('memories', 'semantic_hash')
    op.drop_column('memories', 'content_hash')
    
    logger.info("✅ Downgrade completed")
