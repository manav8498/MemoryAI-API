"""
Comprehensive Test Script for User Profile System

Tests:
1. Automatic profile extraction from memories
2. GET /v1/profile endpoint
3. Manual fact addition
4. Profile + search combo
5. Profile snapshots and history
6. Profile stats and cleanup
"""
import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx
from backend.core.config import settings
from backend.core.logging_config import logger


# Test configuration
API_BASE = "http://localhost:8000"
TEST_USER_EMAIL = "profile_test@example.com"
TEST_USER_PASSWORD = "testpassword123"


class ProfileSystemTester:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=API_BASE, timeout=30.0)
        self.token = None
        self.user_id = None
        self.collection_id = None
        self.memory_id = None

    async def cleanup(self):
        """Clean up test client"""
        await self.client.aclose()

    async def register_and_login(self):
        """Register test user and login"""
        logger.info("🔐 Registering and logging in test user...")

        # Try to register (might fail if user exists)
        try:
            response = await self.client.post("/v1/auth/register", json={
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD,
                "full_name": "Profile Test User"
            })
            if response.status_code == 201:
                logger.info("✓ User registered")
            else:
                logger.info(f"⚠ User might already exist (status {response.status_code})")
        except Exception as e:
            logger.warning(f"Registration failed (user might exist): {e}")

        # Login
        response = await self.client.post("/v1/auth/login", json={
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        })

        if response.status_code != 200:
            logger.error(f"❌ Login failed: {response.text}")
            return False

        data = response.json()
        self.token = data["access_token"]
        self.user_id = data["user_id"]

        logger.info(f"✅ Logged in successfully! User ID: {self.user_id}")
        return True

    async def create_collection(self):
        """Create a test collection"""
        logger.info("📁 Creating test collection...")

        response = await self.client.post(
            "/v1/collections",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "name": "Profile Test Collection",
                "description": "Test collection for profile extraction"
            }
        )

        if response.status_code != 201:
            logger.error(f"❌ Collection creation failed: {response.text}")
            return False

        self.collection_id = response.json()["id"]
        logger.info(f"✅ Collection created: {self.collection_id}")
        return True

    async def create_memory_with_profile_facts(self):
        """Create a memory that should extract profile facts"""
        logger.info("💾 Creating memory with profile-relevant content...")

        # Rich content that should extract multiple profile facts
        content = """
        I'm a Senior Software Engineer at Google working on cloud infrastructure.
        I have 8 years of experience in Python, Go, and distributed systems.

        Currently working on a new Memory AI API project using FastAPI and PostgreSQL.
        I prefer VSCode over other IDEs and use vim keybindings.

        I graduated from Stanford with a CS degree in 2015.
        My goal is to build AI-powered developer tools that enhance productivity.

        Recently learned about vector databases and semantic search.
        I'm interested in LLMs, RAG systems, and memory augmentation for AI.
        """

        response = await self.client.post(
            "/v1/memories",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "content": content,
                "collection_id": self.collection_id,
                "importance": 0.8,
                "custom_metadata": {
                    "test": "profile_extraction"
                }
            }
        )

        if response.status_code != 201:
            logger.error(f"❌ Memory creation failed: {response.text}")
            return False

        self.memory_id = response.json()["id"]
        logger.info(f"✅ Memory created: {self.memory_id}")

        # Wait a bit for async profile extraction to complete
        logger.info("⏳ Waiting 3 seconds for async profile extraction...")
        await asyncio.sleep(3)

        return True

    async def test_get_profile(self):
        """Test GET /v1/profile endpoint"""
        logger.info("\n🧪 TEST 1: GET /v1/profile endpoint")

        response = await self.client.post(
            "/v1/profile",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "include_static": True,
                "include_dynamic": True,
                "min_confidence": 0.0
            }
        )

        if response.status_code != 200:
            logger.error(f"❌ Profile retrieval failed: {response.text}")
            return False

        profile_data = response.json()

        logger.info(f"✅ Profile retrieved successfully!")
        logger.info(f"   Static facts: {len(profile_data['profile']['static'])}")
        logger.info(f"   Dynamic facts: {len(profile_data['profile']['dynamic'])}")

        # Show extracted facts
        if profile_data['profile']['static']:
            logger.info("\n   📊 Static Facts:")
            for fact in profile_data['profile']['static'][:5]:
                logger.info(f"      • {fact['key']}: {fact['value']} (confidence: {fact['confidence']})")

        if profile_data['profile']['dynamic']:
            logger.info("\n   📊 Dynamic Facts:")
            for fact in profile_data['profile']['dynamic'][:5]:
                logger.info(f"      • {fact['key']}: {fact['value']} (confidence: {fact['confidence']})")

        return True

    async def test_add_manual_fact(self):
        """Test manually adding a profile fact"""
        logger.info("\n🧪 TEST 2: Manually add profile fact")

        response = await self.client.post(
            "/v1/profile/facts",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "fact_key": "preferred_language",
                "fact_value": "Python for backend, TypeScript for frontend",
                "profile_type": "static",
                "category": "preference",
                "confidence": 1.0,
                "importance": 0.9
            }
        )

        if response.status_code != 201:
            logger.error(f"❌ Manual fact addition failed: {response.text}")
            return False

        result = response.json()
        logger.info(f"✅ Manual fact added: {result['fact_key']}")
        return True

    async def test_profile_with_search(self):
        """Test profile + search combo"""
        logger.info("\n🧪 TEST 3: Profile + search combo")

        response = await self.client.post(
            "/v1/profile",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "q": "software engineering",
                "collection_id": self.collection_id,
                "search_limit": 5,
                "include_static": True,
                "include_dynamic": True
            }
        )

        if response.status_code != 200:
            logger.error(f"❌ Profile + search failed: {response.text}")
            return False

        data = response.json()

        logger.info(f"✅ Profile + search successful!")
        logger.info(f"   Profile facts: {data['metadata']['profile_facts']}")
        logger.info(f"   Search results: {data['metadata']['search_results']}")

        if data['searchResults']:
            logger.info(f"\n   🔍 Search Results:")
            for result in data['searchResults']['results'][:3]:
                logger.info(f"      • {result['content'][:80]}...")

        return True

    async def test_get_all_facts(self):
        """Test getting all facts with filters"""
        logger.info("\n🧪 TEST 4: Get all facts with filters")

        response = await self.client.get(
            "/v1/profile/facts?profile_type=static&min_confidence=0.5",
            headers={"Authorization": f"Bearer {self.token}"}
        )

        if response.status_code != 200:
            logger.error(f"❌ Get facts failed: {response.text}")
            return False

        data = response.json()
        logger.info(f"✅ Retrieved {len(data['static'])} static facts")
        return True

    async def test_profile_snapshot(self):
        """Test creating profile snapshot"""
        logger.info("\n🧪 TEST 5: Create profile snapshot")

        response = await self.client.post(
            "/v1/profile/snapshot",
            headers={"Authorization": f"Bearer {self.token}"}
        )

        if response.status_code != 200:
            logger.error(f"❌ Snapshot creation failed: {response.text}")
            return False

        data = response.json()
        logger.info(f"✅ Snapshot created: {data['snapshot_id']}")
        return True

    async def test_profile_history(self):
        """Test getting profile history"""
        logger.info("\n🧪 TEST 6: Get profile history")

        response = await self.client.get(
            "/v1/profile/history?limit=10",
            headers={"Authorization": f"Bearer {self.token}"}
        )

        if response.status_code != 200:
            logger.error(f"❌ History retrieval failed: {response.text}")
            return False

        data = response.json()
        logger.info(f"✅ Retrieved {data['total']} history operations")

        if data.get('history'):
            logger.info("\n   📜 Recent Operations:")
            for op in data['history'][:5]:
                new_value = op.get('new_value') or 'N/A'
                logger.info(f"      • {op['operation_type']}: {new_value[:50]}")

        return True

    async def test_profile_stats(self):
        """Test getting profile stats"""
        logger.info("\n🧪 TEST 7: Get profile stats")

        response = await self.client.get(
            "/v1/profile/stats",
            headers={"Authorization": f"Bearer {self.token}"}
        )

        if response.status_code != 200:
            logger.error(f"❌ Stats retrieval failed: {response.text}")
            return False

        data = response.json()['stats']
        logger.info(f"✅ Profile Stats:")
        logger.info(f"   Total facts: {data['total_facts']}")
        logger.info(f"   Static facts: {data['static_facts']}")
        logger.info(f"   Dynamic facts: {data['dynamic_facts']}")
        logger.info(f"   Avg confidence: {data['average_confidence']:.2f}")
        logger.info(f"   Categories: {data['categories']}")

        return True

    async def test_delete_fact(self):
        """Test deleting a profile fact"""
        logger.info("\n🧪 TEST 8: Delete profile fact")

        response = await self.client.delete(
            "/v1/profile/facts/preferred_language",
            headers={"Authorization": f"Bearer {self.token}"}
        )

        if response.status_code != 200:
            logger.error(f"❌ Fact deletion failed: {response.text}")
            return False

        logger.info(f"✅ Fact deleted successfully")
        return True

    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting User Profile System Tests\n")
        logger.info("=" * 60)

        results = {}

        # Setup
        if not await self.register_and_login():
            logger.error("Failed to setup test user")
            return results

        if not await self.create_collection():
            logger.error("Failed to create test collection")
            return results

        if not await self.create_memory_with_profile_facts():
            logger.error("Failed to create test memory")
            return results

        # Run tests
        tests = [
            ("Get Profile", self.test_get_profile),
            ("Add Manual Fact", self.test_add_manual_fact),
            ("Profile + Search", self.test_profile_with_search),
            ("Get All Facts", self.test_get_all_facts),
            ("Create Snapshot", self.test_profile_snapshot),
            ("Get History", self.test_profile_history),
            ("Get Stats", self.test_profile_stats),
            ("Delete Fact", self.test_delete_fact),
        ]

        for test_name, test_func in tests:
            try:
                results[test_name] = await test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")

        logger.info(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")

        if passed == total:
            logger.info("\n🎉 All tests passed! User Profile system is working correctly!")
        else:
            logger.warning(f"\n⚠️  {total - passed} tests failed. Check logs above for details.")

        return results


async def main():
    """Main test runner"""
    tester = ProfileSystemTester()

    try:
        results = await tester.run_all_tests()

        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)

    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
