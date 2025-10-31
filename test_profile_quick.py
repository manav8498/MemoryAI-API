#!/usr/bin/env python3
"""Quick test to verify profile system is working"""
import asyncio
import httpx

API_BASE = "http://localhost:8000"

async def test_profile_system():
    """Test profile system functionality"""
    print("🧪 Testing Profile System Integration\n")
    print("=" * 60)

    # Step 1: Register user
    print("\n1. Registering new user...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE}/v1/auth/register",
            json={
                "email": f"quicktest_{asyncio.get_event_loop().time()}@example.com",
                "password": "testpass123",
                "full_name": "Quick Test User"
            }
        )

        if response.status_code != 201:
            print(f"❌ Registration failed: {response.text}")
            return False

        data = response.json()
        token = data["access_token"]
        user_id = data["user_id"]
        print(f"✅ User registered: {user_id}")

        # Step 2: Create collection
        print("\n2. Creating collection...")
        response = await client.post(
            f"{API_BASE}/v1/collections",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "Test Collection",
                "description": "Test collection for profile"
            }
        )

        if response.status_code != 201:
            print(f"❌ Collection creation failed: {response.text}")
            return False

        collection_id = response.json()["id"]
        print(f"✅ Collection created: {collection_id}")

        # Step 3: Test profile endpoint
        print("\n3. Testing GET /v1/profile endpoint...")
        response = await client.post(
            f"{API_BASE}/v1/profile",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "include_static": True,
                "include_dynamic": True
            }
        )

        if response.status_code != 200:
            print(f"❌ Profile endpoint failed: {response.status_code} - {response.text}")
            return False

        profile_data = response.json()
        print(f"✅ Profile endpoint works!")
        print(f"   Static facts: {len(profile_data['profile']['static'])}")
        print(f"   Dynamic facts: {len(profile_data['profile']['dynamic'])}")

        # Step 4: Test conversation starters endpoint
        print("\n4. Testing GET /v1/profile/starters endpoint...")
        response = await client.get(
            f"{API_BASE}/v1/profile/starters?count=3",
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code != 200:
            print(f"❌ Starters endpoint failed: {response.status_code} - {response.text}")
            return False

        starters_data = response.json()
        print(f"✅ Conversation starters endpoint works!")
        print(f"   Generated {starters_data['count']} starters:")
        for starter in starters_data['starters']:
            print(f"   • {starter['question']}")

        # Step 5: Add a manual fact
        print("\n5. Testing manual fact addition...")
        response = await client.post(
            f"{API_BASE}/v1/profile/facts",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "fact_key": "role",
                "fact_value": "Software Engineer",
                "profile_type": "static",
                "category": "role",
                "confidence": 1.0,
                "importance": 0.9
            }
        )

        if response.status_code != 201:
            print(f"❌ Add fact failed: {response.status_code} - {response.text}")
            return False

        print(f"✅ Manual fact added successfully")

        # Step 6: Verify fact was added
        print("\n6. Verifying profile updates...")
        response = await client.post(
            f"{API_BASE}/v1/profile",
            headers={"Authorization": f"Bearer {token}"},
            json={}
        )

        if response.status_code != 200:
            print(f"❌ Profile retrieval failed")
            return False

        profile_data = response.json()
        print(f"✅ Profile now has {len(profile_data['profile']['static'])} static facts")

        if profile_data['profile']['static']:
            print("   Facts:")
            for fact in profile_data['profile']['static']:
                print(f"   • {fact['key']}: {fact['value']}")

    print("\n" + "=" * 60)
    print("🎉 All profile system tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_profile_system())
    exit(0 if success else 1)
