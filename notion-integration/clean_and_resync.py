#!/usr/bin/env python3.11
"""
Clean old Untitled memories and re-sync all Notion pages with proper titles
"""
import asyncio
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

MEMORY_AI_API_KEY = os.getenv("MEMORY_AI_API_KEY")
MEMORY_AI_BASE_URL = os.getenv("MEMORY_AI_BASE_URL", "http://localhost:8000")
COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"

async def clean_and_resync():
    """Delete all old memories and resync from Notion"""

    print("\n" + "="*70)
    print("CLEANING OLD MEMORIES AND RE-SYNCING NOTION")
    print("="*70 + "\n")

    client = httpx.AsyncClient(
        base_url=MEMORY_AI_BASE_URL,
        headers={"Authorization": f"Bearer {MEMORY_AI_API_KEY}"},
        timeout=60.0
    )

    try:
        # Step 1: Get all memories in the collection
        print("🔍 Step 1: Fetching all existing memories...\n")

        # Search for all memories
        response = await client.post(
            "/v1/search",
            json={
                "query": "application",  # Generic query to get all
                "collection_id": COLLECTION_ID,
                "limit": 200  # Get as many as possible
            }
        )

        if response.status_code != 200:
            print(f"❌ Error fetching memories: {response.status_code}")
            print(response.text)
            return

        results = response.json().get("results", [])
        memory_ids = [r.get("id") for r in results if r.get("id")]

        print(f"✓ Found {len(memory_ids)} memories to delete\n")

        # Step 2: Delete all old memories
        if memory_ids:
            print("🗑️  Step 2: Deleting old memories...\n")

            deleted_count = 0
            for i, memory_id in enumerate(memory_ids, 1):
                try:
                    del_response = await client.delete(f"/v1/memories/{memory_id}")
                    if del_response.status_code in [200, 204]:
                        deleted_count += 1
                        if i % 10 == 0:
                            print(f"   Deleted {deleted_count}/{len(memory_ids)} memories...")
                except Exception as e:
                    print(f"   ⚠ Failed to delete {memory_id}: {e}")

            print(f"\n✓ Deleted {deleted_count} old memories\n")

        # Step 3: Re-sync from Notion with fixed code
        print("🔄 Step 3: Re-syncing all Notion pages with correct parsing...\n")

        # Import and run auto_sync_all
        from notion_memory_bot import NotionMemoryBot

        bot = NotionMemoryBot()
        await bot.initialize()

        # Discover and sync all databases
        try:
            db_response = await bot.notion.search(
                filter={"property": "object", "value": "database"}
            )
            databases = db_response["results"]

            print(f"✓ Found {len(databases)} databases to sync\n")

            total_synced = 0
            for db in databases:
                db_id = db["id"]
                db_title = "Unknown"

                # Get database name
                if "title" in db and len(db["title"]) > 0:
                    db_title = "".join([t.get("plain_text", "") for t in db["title"]])

                print(f"\n📚 Syncing database: {db_title}")

                # Query all pages
                try:
                    pages_response = await bot.notion.databases.query(database_id=db_id)
                    pages = pages_response["results"]

                    print(f"   Found {len(pages)} pages")

                    synced_count = 0
                    for page in pages:
                        page_id = page["id"]
                        memory_id = await bot.sync_page_to_memory(page_id)
                        if memory_id:
                            synced_count += 1

                        # Rate limiting
                        await asyncio.sleep(0.3)

                    total_synced += synced_count
                    print(f"   ✓ Synced {synced_count}/{len(pages)} pages")

                except Exception as e:
                    print(f"   ⚠ Skipped database: {e}")

            print(f"\n✅ Total synced: {total_synced} pages")

        except Exception as e:
            print(f"❌ Error syncing: {e}")
            import traceback
            traceback.print_exc()

        await bot.close()

    finally:
        await client.aclose()

    print("\n" + "="*70)
    print("✅ CLEANUP AND RE-SYNC COMPLETE!")
    print("="*70 + "\n")
    print("Next steps:")
    print("1. Test search: python3.11 search.py 'your query'")
    print("2. Chat with bot: python3.11 chat.py")
    print("3. Test Poe bot at http://localhost:8080\n")

if __name__ == "__main__":
    asyncio.run(clean_and_resync())
