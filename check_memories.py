#!/usr/bin/env python3.11
"""Check how many memories are in Memory AI"""
import httpx
import asyncio

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
BASE_URL = "http://localhost:8000"
COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"

async def check():
    async with httpx.AsyncClient() as client:
        # Search with a generic query
        response = await client.post(
            f"{BASE_URL}/v1/search",
            json={
                "query": "application",
                "collection_id": COLLECTION_ID,
                "limit": 100
            },
            headers={"Authorization": f"Bearer {TOKEN}"}
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"Total results found: {len(results)}")

            # Get unique titles
            titles = {}
            for r in results:
                title = r.get("metadata", {}).get("title", "Untitled")
                if title not in titles:
                    titles[title] = r

            print(f"Unique pages: {len(titles)}")
            print("\nPages:")
            for i, title in enumerate(titles.keys(), 1):
                print(f"{i}. {title}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    asyncio.run(check())
