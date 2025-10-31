#!/usr/bin/env python3.11
"""Test complete metadata flow"""
import httpx
import asyncio
import json

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
BASE_URL = "http://localhost:8000"
COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"

async def test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create memory with metadata
        print("📝 Step 1: Creating memory with custom metadata...\n")

        create_payload = {
            "collection_id": COLLECTION_ID,
            "content": "# Test Memory With Custom Title\n\nThis is a test memory with custom metadata including a title.",
            "metadata": {
                "title": "Test Memory With Custom Title",
                "source": "test_script",
                "custom_field": "test_value"
            }
        }

        create_response = await client.post(
            f"{BASE_URL}/v1/memories",
            json=create_payload,
            headers={"Authorization": f"Bearer {TOKEN}"}
        )

        if create_response.status_code not in [200, 201]:
            print(f"❌ Failed to create memory: {create_response.status_code}")
            print(create_response.text)
            return

        memory = create_response.json()
        memory_id = memory.get("id")
        print(f"✓ Memory created: {memory_id}\n")

        # Step 2: Search for it
        print("🔍 Step 2: Searching for the memory...\n")

        await asyncio.sleep(2)  # Wait for indexing

        search_response = await client.post(
            f"{BASE_URL}/v1/search",
            json={
                "query": "test memory custom title",
                "collection_id": COLLECTION_ID,
                "limit": 10
            },
            headers={"Authorization": f"Bearer {TOKEN}"}
        )

        if search_response.status_code != 200:
            print(f"❌ Search failed: {search_response.status_code}")
            print(search_response.text)
            return

        results = search_response.json().get("results", [])
        print(f"Found {len(results)} results\n")

        # Find our test memory
        for i, result in enumerate(results, 1):
            if result.get("memory_id") == memory_id:
                print(f"✓ Found our test memory at position {i}")
                print(f"  Memory ID: {result.get('memory_id')}")
                print(f"  Metadata: {json.dumps(result.get('metadata', {}), indent=4)}")
                print(f"  Title in metadata: {result.get('metadata', {}).get('title', 'NOT FOUND')}")
                print(f"  Source in metadata: {result.get('metadata', {}).get('source', 'NOT FOUND')}")
                print(f"  Custom field: {result.get('metadata', {}).get('custom_field', 'NOT FOUND')}")
                print()

                if result.get('metadata', {}).get('title') == "Test Memory With Custom Title":
                    print("✅ SUCCESS! Metadata is being returned correctly!")
                else:
                    print("❌ FAIL! Metadata not found in search results")
                break
        else:
            print(f"⚠ Test memory not found in search results")

        # Step 3: Clean up
        print("\n🗑️  Step 3: Cleaning up test memory...")
        delete_response = await client.delete(
            f"{BASE_URL}/v1/memories/{memory_id}",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        if delete_response.status_code in [200, 204]:
            print("✓ Test memory deleted\n")

if __name__ == "__main__":
    asyncio.run(test())
