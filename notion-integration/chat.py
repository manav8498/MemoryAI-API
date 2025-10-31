#!/usr/bin/env python3.11
"""
Chat with ALL your Notion data
Ask anything - searches across all synced pages automatically
"""
import asyncio
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot

load_dotenv()

async def chat_with_notion():
    """Interactive chat with your Notion data"""

    print("\n" + "="*60)
    print("💬 CHAT WITH YOUR NOTION DATA")
    print("="*60 + "\n")

    bot = NotionMemoryBot()
    await bot.initialize()

    # Check how many memories we have
    try:
        response = await bot.memory_client.get(
            f"/v1/collections/{bot.collection_id}"
        )
        memory_count = response.json().get("memory_count", 0)
        print(f"📊 Connected to {memory_count} Notion pages/databases\n")

        if memory_count == 0:
            print("⚠️  No Notion data synced yet!")
            print("   Run: python3.11 auto_sync_all.py first\n")
            await bot.close()
            return

    except Exception as e:
        print(f"⚠️  Could not check memories: {e}\n")

    print("Ask me anything about your Notion data!")
    print("Examples:")
    print('  - "What jobs have I applied to?"')
    print('  - "Show me all interviews"')
    print('  - "When did I apply to GoMake?"')
    print('  - "What companies are in review status?"')
    print("")
    print("Type 'quit' or 'exit' to stop\n")
    print("="*60 + "\n")

    while True:
        try:
            # Get user question
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            # Search across all Notion data
            print("\n🔍 Searching across all your Notion pages...\n")

            results = await bot.search_memories(question, top_k=10)

            if not results:
                print("❌ No relevant information found.\n")
                continue

            # Display results
            print(f"✓ Found {len(results)} relevant pages:\n")
            print("-"*60 + "\n")

            for i, result in enumerate(results[:5], 1):  # Show top 5
                metadata = result.get("metadata", {})
                title = metadata.get("title", "Unknown")
                score = result.get("score", 0)
                content = result.get("content", "")

                # Extract key info from content
                lines = content.split("\n")
                preview_lines = []

                # Get title and first few properties
                for line in lines[:8]:
                    if line.strip():
                        preview_lines.append(line)

                preview = "\n   ".join(preview_lines)

                print(f"📄 Result {i}: {title}")
                print(f"   Relevance: {score:.3f}")
                print(f"   {preview}")
                print("")

            print("-"*60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    await bot.close()

if __name__ == "__main__":
    asyncio.run(chat_with_notion())
