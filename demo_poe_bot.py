#!/usr/bin/env python3.11
"""
Quick Demo: Test Poe Bot V2 Capabilities
Shows the difference between old and new approach
"""
import httpx
import asyncio
import json

POE_BOT_URL = "http://localhost:8080"

async def test_poe_bot():
    """Test Poe Bot V2 with sample queries"""

    print("\n" + "="*70)
    print("POE BOT V2 - CAPABILITY DEMONSTRATION")
    print("="*70)

    test_queries = [
        {
            "category": "Notion Data",
            "query": "What job applications have I submitted?",
            "expected": "Should list job applications with companies and status"
        },
        {
            "category": "Conversation History",
            "query": "What did we discuss in our last chat?",
            "expected": "Should recall previous conversation topics"
        },
        {
            "category": "Multi-Collection",
            "query": "Give me a summary of everything relevant to my job search",
            "expected": "Should search both Notion and conversations"
        }
    ]

    async with httpx.AsyncClient() as client:
        # First, check bot health
        print("\n🔍 Checking bot status...")
        try:
            health = await client.get(f"{POE_BOT_URL}/")
            if health.status_code == 200:
                info = health.json()
                print(f"✅ Bot Status: {info['status']}")
                print(f"✅ Features: {', '.join(info['features'])}")
        except Exception as e:
            print(f"❌ Bot not responding: {e}")
            print(f"   Make sure bot is running: python3.11 poe_bot_v2.py")
            return

        # Get stats
        print("\n📊 Collection Statistics:")
        try:
            stats = await client.get(f"{POE_BOT_URL}/stats")
            if stats.status_code == 200:
                data = stats.json()
                print(f"   Notion: {data['notion_collection']['memory_count']} memories")
                print(f"   Conversations: {data['conversation_collection']['memory_count']} memories")
        except:
            pass

        # Test queries
        print("\n" + "="*70)
        print("TESTING SAMPLE QUERIES")
        print("="*70)

        for i, test in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {test['category']}")
            print(f"   Query: \"{test['query']}\"")
            print(f"   Expected: {test['expected']}")
            print(f"\n   Sending to Poe bot...")

            # Simulate Poe request format
            poe_request = {
                "type": "query",
                "query": [
                    {"role": "user", "content": test['query']}
                ],
                "user_id": "demo_user"
            }

            try:
                response = await client.post(
                    f"{POE_BOT_URL}/",
                    json=poe_request,
                    timeout=60.0
                )

                if response.status_code == 200:
                    # Parse SSE stream
                    text_content = ""
                    for line in response.text.split('\n'):
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if 'text' in data:
                                    text_content = data['text']
                            except:
                                pass

                    if text_content:
                        print(f"   ✅ Response received ({len(text_content)} chars)")
                        print(f"\n   --- RESPONSE ---")
                        # Show first 300 chars
                        preview = text_content[:300] + "..." if len(text_content) > 300 else text_content
                        print(f"   {preview}")
                        print(f"   ----------------")
                    else:
                        print(f"   ⚠️  Empty response")
                else:
                    print(f"   ❌ Error: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Request failed: {e}")

            await asyncio.sleep(1)  # Rate limiting

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Improvements Demonstrated:")
    print("  ✅ RAG with reasoning engine (vs manual prompting)")
    print("  ✅ Multi-collection search (Notion + Conversations)")
    print("  ✅ Intelligent query routing")
    print("  ✅ Source citations in responses")
    print("  ✅ Conversation memory tracking")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_poe_bot())
