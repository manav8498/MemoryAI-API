#!/usr/bin/env python3.11
"""Check database directly for metadata"""
import asyncio
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from backend.core.database import get_db_session
from backend.models.memory import Memory, MemoryMetadata

async def check():
    async for db in get_db_session():
        # Get a few memories from Notion Notes collection
        result = await db.execute(
            select(Memory)
            .where(Memory.collection_id == "32ab4192-241a-48fc-9051-6829246b0ca7")
            .options(selectinload(Memory.extended_metadata))
            .limit(5)
        )
        memories = result.scalars().all()

        print(f"Found {len(memories)} memories\n")

        for i, memory in enumerate(memories, 1):
            print(f"Memory {i}:")
            print(f"  ID: {memory.id}")
            print(f"  Content preview: {memory.content[:80]}...")
            print(f"  Has extended_metadata: {memory.extended_metadata is not None}")

            if memory.extended_metadata:
                print(f"  Custom metadata: {memory.extended_metadata.custom_metadata}")
            else:
                print(f"  Custom metadata: NONE")

                # Check if metadata exists but wasn't loaded
                meta_result = await db.execute(
                    select(MemoryMetadata).where(MemoryMetadata.memory_id == memory.id)
                )
                meta = meta_result.scalar_one_or_none()
                print(f"  Direct query for metadata: {meta is not None}")
                if meta:
                    print(f"    Found metadata: {meta.custom_metadata}")
            print()

        break

if __name__ == "__main__":
    asyncio.run(check())
