#!/usr/bin/env python3.11
"""
Simple script to sync a single Notion page to Memory AI
"""
import asyncio
import sys
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot

load_dotenv()

async def sync_page(page_id: str):
    """Sync a single Notion page"""

    print("\n" + "="*60)
    print("SYNCING NOTION PAGE TO MEMORY AI")
    print("="*60 + "\n")

    # Initialize bot
    print("🚀 Initializing bot...")
    bot = NotionMemoryBot()
    await bot.initialize()
    print("✓ Bot ready!\n")

    # Sync page
    print(f"📄 Syncing page: {page_id[:8]}...\n")
    memory_id = await bot.sync_page_to_memory(page_id)

    if memory_id:
        print("\n" + "="*60)
        print("✅ PAGE SYNCED SUCCESSFULLY!")
        print("="*60)
        print(f"\nMemory ID: {memory_id}")
        print("\nYou can now:")
        print("  • Search for this page")
        print("  • Ask questions about it")
        print(f'  • Try: python3.11 ask.py "What is this page about?"')
        print("")
    else:
        print("\n" + "="*60)
        print("❌ SYNC FAILED")
        print("="*60)
        print("\nPossible reasons:")
        print("  • Page is empty (no content)")
        print("  • Page not shared with integration")
        print("  • Invalid page ID")
        print("")

    await bot.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Error: Page ID required")
        print("\nUsage:")
        print("  python3.11 sync_page.py <page_id>")
        print("\nHow to get page ID:")
        print("  1. Open any Notion page")
        print("  2. Click 'Share' → 'Copy link'")
        print("  3. URL: https://notion.so/Page-Name-abc123...?v=...")
        print("  4. Page ID = abc123... (part after page name)")
        print("")
        sys.exit(1)

    page_id = sys.argv[1].strip()

    # Clean up page ID (remove URL parts if user pasted full URL)
    if "?" in page_id:
        page_id = page_id.split("?")[0]
    if "/" in page_id:
        page_id = page_id.split("/")[-1]
    if "-" in page_id:
        # Extract just the ID part after last dash
        parts = page_id.split("-")
        if len(parts[-1]) == 32:  # Standard Notion ID length
            page_id = parts[-1]

    asyncio.run(sync_page(page_id))
