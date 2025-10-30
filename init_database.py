#!/usr/bin/env python3
"""
Initialize the database schema using SQLAlchemy models.
Run this before applying migrations.
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.database import init_db


async def main():
    """Initialize database tables."""
    print("🔧 Initializing database schema...")
    try:
        await init_db()
        print("✅ Database schema initialized successfully!")
        print("\nNext steps:")
        print("1. Run migrations:")
        print("   docker exec -i memory-postgres psql -U memory_ai -d memory_ai_db < backend/migrations/001_add_procedural_memory.sql")
        print("   docker exec -i memory-postgres psql -U memory_ai -d memory_ai_db < backend/migrations/002_add_rl_trajectories.sql")
        print("\n2. Verify tables:")
        print("   docker exec -it memory-postgres psql -U memory_ai -d memory_ai_db -c '\\dt'")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

