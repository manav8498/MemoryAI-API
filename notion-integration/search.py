#!/usr/bin/env python3.11
"""
Simple script to search your Notion memories
"""
import asyncio
import sys
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot

load_dotenv()

async def search_memories(query: str, top_k: int = 10):
    """Search your Notion memories"""

    print("\n" + "="*60)
    print("SEARCHING YOUR NOTION MEMORIES")
    print("="*60 + "\n")

    # Initialize bot
    bot = NotionMemoryBot()
    await bot.initialize()

    # Search
    print(f"🔍 Searching for: {query}\n")
    results = await bot.search_memories(query, top_k=top_k)

    print("="*60)
    print(f"FOUND {len(results)} RESULTS")
    print("="*60 + "\n")

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        title = metadata.get("title", "Unknown")
        score = result.get("score", 0)
        content_preview = result.get("content", "")[:200]

        print(f"{i}. {title}")
        print(f"   Score: {score:.3f}")
        print(f"   Preview: {content_preview}...")
        print("")

    await bot.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Error: Search query required")
        print("\nUsage:")
        print('  python3.11 search.py "your query"')
        print("\nExamples:")
        print('  python3.11 search.py "GoMake"')
        print('  python3.11 search.py "interview"')
        print('  python3.11 search.py "April 2025"')
        print("")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    asyncio.run(search_memories(query))
