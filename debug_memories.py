#!/usr/bin/env python3.11
"""Debug what memories are actually stored"""
import httpx
import asyncio
import json

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
BASE_URL = "http://localhost:8000"

async def debug():
    async with httpx.AsyncClient() as client:
        # Get collections
        print("📚 Getting collections...\n")
        response = await client.get(
            f"{BASE_URL}/v1/collections",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )

        if response.status_code == 200:
            collections = response.json()
            for col in collections:
                print(f"Collection: {col['name']} (ID: {col['id']})")

        # Try to get memories directly
        print("\n🔍 Getting memories from Notion Notes collection...\n")

        # Search with different queries
        for query in ["application", "job", "interview", "notion labs"]:
            response = await client.post(
                f"{BASE_URL}/v1/search",
                json={
                    "query": query,
                    "collection_id": "32ab4192-241a-48fc-9051-6829246b0ca7",
                    "limit": 5
                },
                headers={"Authorization": f"Bearer {TOKEN}"}
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print(f"Query '{query}': {len(results)} results")
                for i, r in enumerate(results[:3], 1):
                    title = r.get("metadata", {}).get("title", "NO TITLE IN METADATA")
                    content_preview = r.get("content", "")[:100]
                    print(f"  {i}. Title in metadata: '{title}'")
                    print(f"     Content starts with: '{content_preview[:60]}...'")
                print()

if __name__ == "__main__":
    asyncio.run(debug())
