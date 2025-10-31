#!/usr/bin/env python3.11
"""
Debug script to see what's in database pages
"""
import asyncio
import sys
import json
from dotenv import load_dotenv
from notion_client import AsyncClient as NotionClient
import os

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")

async def debug_database(database_id: str):
    """Debug what's in database pages"""

    notion = NotionClient(auth=NOTION_TOKEN)

    print("\n" + "="*60)
    print("DEBUGGING NOTION DATABASE PAGES")
    print("="*60 + "\n")

    # Query database
    print(f"📚 Database ID: {database_id}\n")
    response = await notion.databases.query(database_id=database_id)
    pages = response["results"]

    print(f"Found {len(pages)} pages in database\n")

    # Debug first 3 pages
    for idx, page in enumerate(pages[:3], 1):
        page_id = page["id"]

        print("\n" + "="*60)
        print(f"PAGE {idx} - ID: {page_id}")
        print("="*60 + "\n")

        # Show properties
        print("📝 Page Properties:")
        for prop_name, prop_value in page["properties"].items():
            print(f"\n  {prop_name}:")
            print(f"    Type: {prop_value['type']}")

            # Extract value based on type
            if prop_value['type'] == 'title':
                title_parts = prop_value.get('title', [])
                text = "".join([t.get('plain_text', '') for t in title_parts])
                print(f"    Value: '{text}'")
            elif prop_value['type'] == 'rich_text':
                text_parts = prop_value.get('rich_text', [])
                text = "".join([t.get('plain_text', '') for t in text_parts])
                print(f"    Value: '{text}'")
            elif prop_value['type'] == 'select':
                select_val = prop_value.get('select')
                print(f"    Value: {select_val.get('name') if select_val else 'None'}")
            elif prop_value['type'] == 'multi_select':
                options = prop_value.get('multi_select', [])
                print(f"    Value: {[o.get('name') for o in options]}")
            elif prop_value['type'] == 'date':
                date_val = prop_value.get('date')
                print(f"    Value: {date_val}")
            elif prop_value['type'] == 'url':
                url_val = prop_value.get('url')
                print(f"    Value: {url_val}")
            else:
                print(f"    Raw: {json.dumps(prop_value, indent=6)}")

        # Get blocks
        print("\n📦 Page Blocks:")
        try:
            blocks_response = await notion.blocks.children.list(block_id=page_id)
            blocks = blocks_response["results"]

            print(f"  Found {len(blocks)} blocks\n")

            if len(blocks) == 0:
                print("  ⚠️  NO BLOCKS - Page body is empty!")
                print("  (Content might be in properties above)")

            for i, block in enumerate(blocks[:5], 1):  # Show first 5 blocks
                block_type = block["type"]
                has_children = block.get("has_children", False)

                print(f"\n  Block {i}:")
                print(f"    Type: {block_type}")
                print(f"    Has children: {has_children}")

                # Try to extract text
                if block_type in block and "rich_text" in block[block_type]:
                    rich_text = block[block_type]["rich_text"]
                    text = "".join([t.get("plain_text", "") for t in rich_text])
                    print(f"    Text: '{text}'")
                elif block_type in block and "text" in block[block_type]:
                    text = block[block_type]["text"]
                    print(f"    Text: '{text}'")
                else:
                    print(f"    Structure: {json.dumps(block[block_type] if block_type in block else {}, indent=6)}")

            if len(blocks) > 5:
                print(f"\n  ... and {len(blocks) - 5} more blocks")

        except Exception as e:
            print(f"  ❌ Error fetching blocks: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "-"*60)

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3.11 debug_database.py <database_id>")
        sys.exit(1)

    database_id = sys.argv[1].strip()

    # Clean up database ID
    if "?" in database_id:
        database_id = database_id.split("?")[0]
    if "/" in database_id:
        database_id = database_id.split("/")[-1]
    if "-" in database_id:
        parts = database_id.split("-")
        if len(parts[-1]) == 32:
            database_id = parts[-1]

    asyncio.run(debug_database(database_id))
