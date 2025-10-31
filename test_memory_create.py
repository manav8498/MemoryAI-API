#!/usr/bin/env python3.11
"""Test creating a memory and verifying metadata is stored"""
import httpx
import asyncio
import json

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
BASE_URL = "http://localhost:8000"
COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"

async def test():
    async with httpx.AsyncClient() as client:
        # Create a test memory with metadata
        print("📝 Creating test memory with metadata...\n")

        payload = {
            "collection_id": COLLECTION_ID,
            "content": "# Test Memory Title\n\nThis is test content",
            "metadata": {
                "title": "Test Memory Title",
                "source": "test",
                "custom_field": "custom_value"
            }
        }

        print(f"Sending payload:\n{json.dumps(payload, indent=2)}\n")

        response = await client.post(
            f"{BASE_URL}/v1/memories",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"}
        )

        print(f"Response status: {response.status_code}")
        result = response.json()
        print(f"Response:\n{json.dumps(result, indent=2)}\n")

        if response.status_code in [200, 201]:
            memory_id = result.get("id")
            print(f"✓ Memory created: {memory_id}\n")

            # Now retrieve it and check metadata
            print("🔍 Retrieving memory to verify metadata...\n")

            get_response = await client.get(
                f"{BASE_URL}/v1/memories/{memory_id}",
                headers={"Authorization": f"Bearer {TOKEN}"}
            )

            if get_response.status_code == 200:
                retrieved = get_response.json()
                print(f"Retrieved memory:\n{json.dumps(retrieved, indent=2)}\n")

                metadata = retrieved.get("metadata", {})
                print(f"Metadata field: {metadata}")
                print(f"Title in metadata: {metadata.get('title', 'NOT FOUND')}")

if __name__ == "__main__":
    asyncio.run(test())
