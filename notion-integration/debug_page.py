#!/usr/bin/env python3.11
"""
Debug script to see what's actually in a Notion page
"""
import asyncio
import sys
import json
from dotenv import load_dotenv
from notion_client import AsyncClient as NotionClient
import os

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")

async def debug_page(page_id: str):
    """Debug what's in a Notion page"""

    notion = NotionClient(auth=NOTION_TOKEN)

    print("\n" + "="*60)
    print("DEBUGGING NOTION PAGE")
    print("="*60 + "\n")

    # Get page details
    print(f"📄 Page ID: {page_id}\n")

    try:
        page = await notion.pages.retrieve(page_id=page_id)
        print("✅ Page found!\n")
        print("Page properties:")
        print(json.dumps(page["properties"], indent=2))
        print("\n" + "-"*60 + "\n")

        # Get blocks
        print("📦 Fetching blocks...\n")
        blocks = await notion.blocks.children.list(block_id=page_id)

        print(f"Found {len(blocks['results'])} blocks\n")

        for i, block in enumerate(blocks["results"], 1):
            block_type = block["type"]
            print(f"\nBlock {i}:")
            print(f"  Type: {block_type}")
            print(f"  Has children: {block.get('has_children', False)}")
            print(f"  Full block data:")
            print(json.dumps(block, indent=4))
            print("-"*60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3.11 debug_page.py <page_id>")
        sys.exit(1)

    page_id = sys.argv[1].strip()

    # Clean up page ID
    if "?" in page_id:
        page_id = page_id.split("?")[0]
    if "/" in page_id:
        page_id = page_id.split("/")[-1]
    if "-" in page_id:
        parts = page_id.split("-")
        if len(parts[-1]) == 32:
            page_id = parts[-1]

    asyncio.run(debug_page(page_id))
