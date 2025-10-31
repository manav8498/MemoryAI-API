#!/usr/bin/env python3.11
"""
Simple script to sync entire Notion database to Memory AI
"""
import asyncio
import sys
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot

load_dotenv()

async def sync_database(database_id: str):
    """Sync all pages from a Notion database"""

    print("\n" + "="*60)
    print("SYNCING NOTION DATABASE TO MEMORY AI")
    print("="*60 + "\n")

    # Initialize bot
    print("🚀 Initializing bot...")
    bot = NotionMemoryBot()
    await bot.initialize()
    print("✓ Bot ready!\n")

    # Sync database
    print(f"📚 Syncing database: {database_id[:8]}...\n")
    await bot.sync_all_pages_in_database(database_id)

    # Show results
    print("\n" + "="*60)
    print("✅ SYNC COMPLETE!")
    print("="*60)
    print("\nYou can now:")
    print("  • Search your memories")
    print("  • Ask questions about your notes")
    print("  • Run: python3.11 notion_memory_bot.py (option 3 or 4)")
    print("")

    await bot.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Error: Database ID required")
        print("\nUsage:")
        print("  python3.11 sync_database.py <database_id>")
        print("\nHow to get database ID:")
        print("  1. Open database in Notion")
        print("  2. Click 'Share' → 'Copy link'")
        print("  3. URL: https://notion.so/DatabaseName-abc123...?v=...")
        print("  4. Database ID = abc123... (part before ?v=)")
        print("")
        sys.exit(1)

    database_id = sys.argv[1].strip()

    # Clean up database ID (remove URL parts if user pasted full URL)
    if "?" in database_id:
        database_id = database_id.split("?")[0]
    if "/" in database_id:
        database_id = database_id.split("/")[-1]
    if "-" in database_id:
        # Extract just the ID part after last dash
        parts = database_id.split("-")
        if len(parts[-1]) == 32:  # Standard Notion ID length
            database_id = parts[-1]

    asyncio.run(sync_database(database_id))
