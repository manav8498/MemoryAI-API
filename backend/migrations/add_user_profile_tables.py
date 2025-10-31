"""
Database Migration: Add User Profile Tables

This migration adds three new tables for the User Profile feature:
1. user_profile_facts - Stores user profile facts (static and dynamic)
2. profile_operations - Tracks all profile changes for auditing
3. profile_snapshots - Stores profile snapshots for versioning

Usage:
    python -m backend.migrations.add_user_profile_tables
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import text
from backend.core.database import engine
from backend.core.logging_config import logger


async def run_migration():
    """
    Run the migration to add user profile tables.
    """
    try:
        async with engine.begin() as conn:
            logger.info("Starting user profile tables migration...")

            # Create user_profile_facts table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_profile_facts (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    profile_type VARCHAR(20) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    fact_key VARCHAR(200) NOT NULL,
                    fact_value TEXT NOT NULL,
                    confidence FLOAT DEFAULT 0.7,
                    importance FLOAT DEFAULT 0.5,
                    source_memory_ids JSON DEFAULT '[]',
                    extraction_metadata JSON DEFAULT '{}',
                    verified VARCHAR(20) DEFAULT 'auto',
                    access_count VARCHAR(20) DEFAULT '0',
                    last_accessed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, fact_key)
                )
            """))
            logger.info("✓ Created user_profile_facts table")

            # Create indexes on user_profile_facts
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_facts_user_id
                ON user_profile_facts(user_id)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_facts_type
                ON user_profile_facts(profile_type)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_facts_category
                ON user_profile_facts(category)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_facts_confidence
                ON user_profile_facts(confidence)
            """))
            logger.info("✓ Created indexes on user_profile_facts")

            # Create profile_operations table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS profile_operations (
                    id VARCHAR(36) PRIMARY KEY,
                    profile_fact_id VARCHAR(36),
                    user_id VARCHAR(36) NOT NULL,
                    operation_type VARCHAR(20) NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    confidence_change FLOAT,
                    trigger_memory_id VARCHAR(36),
                    trigger_type VARCHAR(50),
                    operation_metadata JSON DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (profile_fact_id) REFERENCES user_profile_facts(id) ON DELETE SET NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """))
            logger.info("✓ Created profile_operations table")

            # Create indexes on profile_operations
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_ops_user_id
                ON profile_operations(user_id)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_ops_fact_id
                ON profile_operations(profile_fact_id)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_ops_created_at
                ON profile_operations(created_at DESC)
            """))
            logger.info("✓ Created indexes on profile_operations")

            # Create profile_snapshots table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS profile_snapshots (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    static_facts JSON DEFAULT '[]',
                    dynamic_facts JSON DEFAULT '[]',
                    snapshot_metadata JSON DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """))
            logger.info("✓ Created profile_snapshots table")

            # Create indexes on profile_snapshots
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_snapshots_user_id
                ON profile_snapshots(user_id)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_profile_snapshots_created_at
                ON profile_snapshots(created_at DESC)
            """))
            logger.info("✓ Created indexes on profile_snapshots")

            logger.info("✅ User profile tables migration completed successfully!")

            # Verify tables were created
            result = await conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('user_profile_facts', 'profile_operations', 'profile_snapshots')
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]

            logger.info(f"Verified tables created: {tables}")

            if len(tables) == 3:
                logger.info("✅ All 3 tables created successfully!")
                return True
            else:
                logger.error(f"❌ Expected 3 tables, found {len(tables)}: {tables}")
                return False

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


async def rollback_migration():
    """
    Rollback the migration (drop user profile tables).
    """
    try:
        async with engine.begin() as conn:
            logger.info("Rolling back user profile tables migration...")

            await conn.execute(text("DROP TABLE IF EXISTS profile_operations CASCADE"))
            logger.info("✓ Dropped profile_operations table")

            await conn.execute(text("DROP TABLE IF EXISTS profile_snapshots CASCADE"))
            logger.info("✓ Dropped profile_snapshots table")

            await conn.execute(text("DROP TABLE IF EXISTS user_profile_facts CASCADE"))
            logger.info("✓ Dropped user_profile_facts table")

            logger.info("✅ Rollback completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        raise


async def check_tables_exist():
    """
    Check if user profile tables already exist.
    """
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('user_profile_facts', 'profile_operations', 'profile_snapshots')
            """))
            tables = [row[0] for row in result]
            return tables

    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return []


async def main():
    """
    Main migration entry point.
    """
    import sys

    # Check if rollback flag is provided
    if "--rollback" in sys.argv:
        logger.info("🔄 Running rollback...")
        await rollback_migration()
        return

    # Check if tables already exist
    existing_tables = await check_tables_exist()

    if existing_tables:
        logger.warning(f"⚠️  Tables already exist: {existing_tables}")
        response = input("Do you want to continue anyway? This may cause errors. (y/N): ")
        if response.lower() != 'y':
            logger.info("Migration cancelled.")
            return

    # Run migration
    logger.info("🚀 Starting migration...")
    success = await run_migration()

    if success:
        logger.info("✅ Migration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Restart your FastAPI server")
        logger.info("2. Test the profile endpoints: POST /v1/profile")
        logger.info("3. Create a memory and check if profile facts are auto-extracted")
    else:
        logger.error("❌ Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
