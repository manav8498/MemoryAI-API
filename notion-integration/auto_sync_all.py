#!/usr/bin/env python3.11
"""
Auto-discover and sync ALL accessible Notion content
"""
import asyncio
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot

load_dotenv()

async def auto_sync_everything():
    """Automatically discover and sync all accessible Notion content"""

    print("\n" + "="*60)
    print("AUTO-SYNCING ALL NOTION CONTENT")
    print("="*60 + "\n")

    bot = NotionMemoryBot()
    await bot.initialize()

    print("🔍 Discovering all accessible databases...\n")

    # Search for all databases
    try:
        response = await bot.notion.search(
            filter={"property": "object", "value": "database"}
        )
        databases = response["results"]

        print(f"✓ Found {len(databases)} accessible databases\n")

        total_synced = 0
        for db in databases:
            db_id = db["id"]
            db_title = "Unknown"

            # Try to get database name
            if "title" in db and len(db["title"]) > 0:
                db_title = "".join([t.get("plain_text", "") for t in db["title"]])

            print(f"\n📚 Syncing database: {db_title}")
            print(f"   ID: {db_id[:8]}...")

            # Query all pages in this database
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
                print(f"   ✓ Synced {synced_count}/{len(pages)} pages from '{db_title}'")

            except Exception as e:
                print(f"   ⚠ Skipped database: {e}")

    except Exception as e:
        print(f"❌ Error discovering databases: {e}")

    # Also search for standalone pages (not in databases)
    print("\n\n🔍 Discovering standalone pages...\n")

    try:
        response = await bot.notion.search(
            filter={"property": "object", "value": "page"}
        )
        standalone_pages = response["results"]

        print(f"✓ Found {len(standalone_pages)} standalone pages\n")

        standalone_synced = 0
        for page in standalone_pages:  # Sync ALL pages
            page_id = page["id"]

            # Get page title
            page_title = "Untitled"
            if "properties" in page:
                for prop_name, prop_value in page["properties"].items():
                    if prop_value.get("type") == "title":
                        title_parts = prop_value.get("title", [])
                        if title_parts:
                            page_title = "".join([t.get("plain_text", "") for t in title_parts])
                        break

            print(f"📄 Syncing: {page_title}")
            memory_id = await bot.sync_page_to_memory(page_id)
            if memory_id:
                standalone_synced += 1

            await asyncio.sleep(0.3)

        total_synced += standalone_synced
        print(f"\n✓ Synced {standalone_synced}/{len(standalone_pages)} standalone pages")

    except Exception as e:
        print(f"⚠ Error with standalone pages: {e}")

    print("\n\n" + "="*60)
    print(f"✅ SYNC COMPLETE - {total_synced} TOTAL PAGES SYNCED")
    print("="*60)
    print("\nYou can now ask questions about ALL your Notion data!")
    print('Run: python3.11 chat.py')
    print("")

    await bot.close()

if __name__ == "__main__":
    asyncio.run(auto_sync_everything())
