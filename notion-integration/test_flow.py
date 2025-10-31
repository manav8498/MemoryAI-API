#!/usr/bin/env python3.11
"""
Simple test to verify Notion → Memory AI integration works
"""
import asyncio
import os
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot

load_dotenv()

async def test_integration():
    print("\n" + "="*60)
    print("TESTING NOTION → MEMORY AI INTEGRATION")
    print("="*60 + "\n")

    # Initialize bot
    print("1️⃣  Initializing bot...")
    bot = NotionMemoryBot()
    await bot.initialize()
    print("   ✅ Bot initialized\n")

    # Test API connection
    print("2️⃣  Testing Memory AI API connection...")
    try:
        response = await bot.memory_client.get("/v1/collections")
        collections = response.json()
        print(f"   ✅ API connected - Found {len(collections)} collections\n")
    except Exception as e:
        print(f"   ❌ API connection failed: {e}\n")
        return

    # Prompt for Notion page ID
    print("3️⃣  Ready to sync Notion page!")
    print("   To get your page ID:")
    print("   - Open any Notion page")
    print("   - Click 'Share' → 'Copy link'")
    print("   - URL looks like: https://notion.so/Page-Name-XXXXX")
    print("   - The XXXXX part is your page ID\n")

    page_id = input("   Enter Notion page ID (or press Enter to skip): ").strip()

    if page_id:
        print(f"\n4️⃣  Syncing page {page_id[:8]}...")
        memory_id = await bot.sync_page_to_memory(page_id)

        if memory_id:
            print(f"   ✅ Page synced! Memory ID: {memory_id[:8]}...\n")

            # Test search
            print("5️⃣  Testing search...")
            results = await bot.search_memories("test", top_k=5)
            print(f"   ✅ Search works - Found {len(results)} results\n")

            # Test question
            print("6️⃣  Testing Q&A...")
            question = "What is this page about?"
            answer = await bot.ask_question(question)
            print(f"   Question: {question}")
            print(f"   Answer: {answer[:200]}...\n")

            print("="*60)
            print("✅ ALL TESTS PASSED!")
            print("="*60)
            print("\nYour Memory AI API is serving the Notion bot correctly! 🎉")
        else:
            print("   ❌ Failed to sync page")
            print("   Make sure:")
            print("   - You've shared the page with your Notion integration")
            print("   - The page has content (not empty)")
    else:
        print("\n   Skipped page sync")
        print("   Your API is working, but you need to sync a page to test Q&A\n")

    await bot.close()

if __name__ == "__main__":
    asyncio.run(test_integration())
