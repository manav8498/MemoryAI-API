#!/usr/bin/env python3.11
"""
Comprehensive Memory AI API Test Suite
Tests ALL capabilities with detailed reporting
"""
import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, Any, List

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
BASE_URL = "http://localhost:8000"
NOTION_COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"
CONVERSATION_COLLECTION_ID = "745a1565-cf04-437d-9cfb-c3b5f075afc6"


class MemoryAITester:
    """Comprehensive test suite for Memory AI API"""

    def __init__(self):
        self.client = None
        self.results = []
        self.test_memory_ids = []

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=30.0
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        print(f"{status}: {name}")
        if details:
            print(f"   {details}")

    async def test_1_health_check(self):
        """Test 1: API Health Check"""
        print("\n" + "="*70)
        print("TEST 1: API Health Check")
        print("="*70)

        try:
            response = await self.client.get("/health")
            passed = response.status_code == 200
            self.log_test(
                "API Health Check",
                passed,
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.log_test("API Health Check", False, f"Error: {e}")

    async def test_2_collections(self):
        """Test 2: List Collections"""
        print("\n" + "="*70)
        print("TEST 2: Collections Management")
        print("="*70)

        try:
            response = await self.client.get("/v1/collections")
            collections = response.json() if response.status_code == 200 else []

            passed = response.status_code == 200 and len(collections) >= 2
            self.log_test(
                "List Collections",
                passed,
                f"Found {len(collections)} collections"
            )

            if collections:
                for col in collections:
                    print(f"   - {col['name']}: {col.get('memory_count', 0)} memories")

        except Exception as e:
            self.log_test("List Collections", False, f"Error: {e}")

    async def test_3_create_memory(self):
        """Test 3: Create Memory with Metadata"""
        print("\n" + "="*70)
        print("TEST 3: Memory Creation")
        print("="*70)

        try:
            test_content = f"""# Test Memory {datetime.now().strftime('%H:%M:%S')}

This is a comprehensive test memory created to verify:
- Content storage
- Metadata handling
- Vector embedding
- Full-text indexing

Created at: {datetime.now().isoformat()}
"""

            payload = {
                "collection_id": NOTION_COLLECTION_ID,
                "content": test_content,
                "metadata": {
                    "title": "Comprehensive Test Memory",
                    "type": "test",
                    "timestamp": datetime.now().isoformat(),
                    "tags": ["test", "comprehensive", "memory_ai"]
                }
            }

            response = await self.client.post("/v1/memories", json=payload)
            passed = response.status_code in [200, 201]

            if passed:
                memory = response.json()
                memory_id = memory.get("id")
                self.test_memory_ids.append(memory_id)
                self.log_test(
                    "Create Memory",
                    True,
                    f"Memory ID: {memory_id[:8]}..."
                )
            else:
                self.log_test(
                    "Create Memory",
                    False,
                    f"Status: {response.status_code}, Error: {response.text[:100]}"
                )

        except Exception as e:
            self.log_test("Create Memory", False, f"Error: {e}")

    async def test_4_vector_search(self):
        """Test 4: Vector Search"""
        print("\n" + "="*70)
        print("TEST 4: Vector Search (Semantic)")
        print("="*70)

        await asyncio.sleep(2)  # Wait for indexing

        try:
            response = await self.client.post(
                "/v1/search",
                json={
                    "query": "comprehensive test memory created for verification",
                    "collection_id": NOTION_COLLECTION_ID,
                    "search_type": "vector",
                    "limit": 10
                }
            )

            passed = response.status_code == 200
            if passed:
                results = response.json().get("results", [])
                self.log_test(
                    "Vector Search",
                    len(results) > 0,
                    f"Found {len(results)} results"
                )

                if results:
                    print(f"   Top result: {results[0].get('metadata', {}).get('title', 'Unknown')}")
                    print(f"   Score: {results[0].get('score', 0):.4f}")
            else:
                self.log_test("Vector Search", False, f"Status: {response.status_code}")

        except Exception as e:
            self.log_test("Vector Search", False, f"Error: {e}")

    async def test_5_hybrid_search(self):
        """Test 5: Hybrid Search (Vector + BM25)"""
        print("\n" + "="*70)
        print("TEST 5: Hybrid Search (Vector + BM25)")
        print("="*70)

        try:
            response = await self.client.post(
                "/v1/search",
                json={
                    "query": "job application interview",
                    "collection_id": NOTION_COLLECTION_ID,
                    "search_type": "hybrid",
                    "limit": 10
                }
            )

            passed = response.status_code == 200
            if passed:
                results = response.json().get("results", [])
                self.log_test(
                    "Hybrid Search",
                    True,
                    f"Found {len(results)} results"
                )

                # Check if metadata is included
                if results:
                    has_metadata = "title" in results[0].get("metadata", {})
                    self.log_test(
                        "Metadata in Search Results",
                        has_metadata,
                        f"Title: {results[0].get('metadata', {}).get('title', 'MISSING')}"
                    )
            else:
                self.log_test("Hybrid Search", False, f"Status: {response.status_code}")

        except Exception as e:
            self.log_test("Hybrid Search", False, f"Error: {e}")

    async def test_6_reasoning_engine(self):
        """Test 6: RAG with Reasoning Engine"""
        print("\n" + "="*70)
        print("TEST 6: RAG with Reasoning Engine")
        print("="*70)

        try:
            response = await self.client.post(
                "/v1/search/reason",
                json={
                    "query": "What job applications have I submitted and what is their current status?",
                    "collection_id": NOTION_COLLECTION_ID,
                    "provider": "gemini",
                    "include_steps": False
                }
            )

            passed = response.status_code == 200
            if passed:
                result = response.json()
                answer = result.get("answer", "")
                sources = result.get("sources", [])

                self.log_test(
                    "Reasoning Engine (RAG)",
                    len(answer) > 50,
                    f"Answer length: {len(answer)} chars, Sources: {len(sources)}"
                )

                print(f"\n   QUESTION: What job applications have I submitted?")
                print(f"   ANSWER: {answer[:200]}...")
                print(f"   SOURCES: {len(sources)} memories used")

                if sources:
                    print(f"\n   Top Sources:")
                    for i, source in enumerate(sources[:3], 1):
                        title = source.get("metadata", {}).get("title", "Unknown")
                        score = source.get("score", 0)
                        print(f"     {i}. {title} (score: {score:.3f})")

            else:
                self.log_test(
                    "Reasoning Engine",
                    False,
                    f"Status: {response.status_code}, Error: {response.text[:100]}"
                )

        except Exception as e:
            self.log_test("Reasoning Engine", False, f"Error: {e}")

    async def test_7_conversation_storage(self):
        """Test 7: Conversation History Storage"""
        print("\n" + "="*70)
        print("TEST 7: Conversation History Storage")
        print("="*70)

        try:
            # Store a conversation turn
            conversation_content = f"""# Test Conversation - {datetime.now().strftime('%H:%M:%S')}

**User:** What is Memory AI?
**Assistant:** Memory AI is an advanced memory management system that uses vector databases, knowledge graphs, and reinforcement learning to provide intelligent memory storage and retrieval.

Timestamp: {datetime.now().isoformat()}
"""

            response = await self.client.post(
                "/v1/memories",
                json={
                    "collection_id": CONVERSATION_COLLECTION_ID,
                    "content": conversation_content,
                    "metadata": {
                        "type": "conversation",
                        "timestamp": datetime.now().isoformat(),
                        "user_query": "What is Memory AI?"
                    }
                }
            )

            passed = response.status_code in [200, 201]
            if passed:
                memory = response.json()
                memory_id = memory.get("id")
                self.test_memory_ids.append(memory_id)
                self.log_test(
                    "Store Conversation",
                    True,
                    f"Conversation ID: {memory_id[:8]}..."
                )
            else:
                self.log_test("Store Conversation", False, f"Status: {response.status_code}")

        except Exception as e:
            self.log_test("Store Conversation", False, f"Error: {e}")

    async def test_8_conversation_retrieval(self):
        """Test 8: Conversation History Retrieval"""
        print("\n" + "="*70)
        print("TEST 8: Conversation History Retrieval")
        print("="*70)

        await asyncio.sleep(1)  # Wait for indexing

        try:
            response = await self.client.post(
                "/v1/search",
                json={
                    "query": "what is memory ai",
                    "collection_id": CONVERSATION_COLLECTION_ID,
                    "search_type": "hybrid",
                    "limit": 5
                }
            )

            passed = response.status_code == 200
            if passed:
                results = response.json().get("results", [])
                self.log_test(
                    "Retrieve Conversation History",
                    len(results) > 0,
                    f"Found {len(results)} conversation turns"
                )
            else:
                self.log_test("Retrieve Conversation History", False, f"Status: {response.status_code}")

        except Exception as e:
            self.log_test("Retrieve Conversation History", False, f"Error: {e}")

    async def test_9_metadata_persistence(self):
        """Test 9: Metadata Persistence"""
        print("\n" + "="*70)
        print("TEST 9: Metadata Persistence")
        print("="*70)

        if not self.test_memory_ids:
            self.log_test("Metadata Persistence", False, "No test memories created")
            return

        try:
            memory_id = self.test_memory_ids[0]

            # Retrieve memory
            response = await self.client.get(f"/v1/memories/{memory_id}")

            passed = response.status_code == 200
            if passed:
                memory = response.json()
                metadata = memory.get("metadata", {})

                has_title = "title" in metadata
                has_type = "type" in metadata
                has_timestamp = "timestamp" in metadata

                all_present = has_title and has_type and has_timestamp

                self.log_test(
                    "Metadata Persistence",
                    all_present,
                    f"Title: {has_title}, Type: {has_type}, Timestamp: {has_timestamp}"
                )

                if metadata:
                    print(f"   Metadata keys: {list(metadata.keys())}")
            else:
                self.log_test("Metadata Persistence", False, f"Status: {response.status_code}")

        except Exception as e:
            self.log_test("Metadata Persistence", False, f"Error: {e}")

    async def test_10_cache_performance(self):
        """Test 10: Cache Performance"""
        print("\n" + "="*70)
        print("TEST 10: Cache Performance")
        print("="*70)

        try:
            query = {
                "query": "job application status",
                "collection_id": NOTION_COLLECTION_ID,
                "limit": 10
            }

            # First request (cache miss)
            import time
            start1 = time.time()
            response1 = await self.client.post("/v1/search", json=query)
            time1 = (time.time() - start1) * 1000

            # Second request (should be cached)
            start2 = time.time()
            response2 = await self.client.post("/v1/search", json=query)
            time2 = (time.time() - start2) * 1000

            passed = response1.status_code == 200 and response2.status_code == 200
            if passed:
                speedup = time1 / time2 if time2 > 0 else 0
                self.log_test(
                    "Cache Performance",
                    speedup > 1.5,  # Cached should be at least 1.5x faster
                    f"First: {time1:.0f}ms, Cached: {time2:.0f}ms, Speedup: {speedup:.1f}x"
                )
            else:
                self.log_test("Cache Performance", False, "Requests failed")

        except Exception as e:
            self.log_test("Cache Performance", False, f"Error: {e}")

    async def test_11_cleanup(self):
        """Test 11: Cleanup Test Data"""
        print("\n" + "="*70)
        print("TEST 11: Cleanup Test Data")
        print("="*70)

        deleted = 0
        for memory_id in self.test_memory_ids:
            try:
                response = await self.client.delete(f"/v1/memories/{memory_id}")
                if response.status_code in [200, 204]:
                    deleted += 1
            except:
                pass

        self.log_test(
            "Cleanup Test Data",
            deleted == len(self.test_memory_ids),
            f"Deleted {deleted}/{len(self.test_memory_ids)} test memories"
        )

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")

        if failed > 0:
            print("\n❌ Failed Tests:")
            for result in self.results:
                if not result["passed"]:
                    print(f"   - {result['name']}: {result['details']}")

        print("\n" + "="*70)

        # Overall status
        if passed == total:
            print("🎉 ALL TESTS PASSED - MEMORY AI IS FULLY OPERATIONAL!")
        elif passed >= total * 0.8:
            print("⚠️ MOST TESTS PASSED - Some issues need attention")
        else:
            print("❌ MANY TESTS FAILED - Critical issues detected")

        print("="*70 + "\n")


async def main():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print("MEMORY AI - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"\nStarting tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {BASE_URL}")
    print(f"Notion Collection: {NOTION_COLLECTION_ID}")
    print(f"Conversation Collection: {CONVERSATION_COLLECTION_ID}")

    async with MemoryAITester() as tester:
        # Run all tests
        await tester.test_1_health_check()
        await tester.test_2_collections()
        await tester.test_3_create_memory()
        await tester.test_4_vector_search()
        await tester.test_5_hybrid_search()
        await tester.test_6_reasoning_engine()
        await tester.test_7_conversation_storage()
        await tester.test_8_conversation_retrieval()
        await tester.test_9_metadata_persistence()
        await tester.test_10_cache_performance()
        await tester.test_11_cleanup()

        # Print summary
        tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
