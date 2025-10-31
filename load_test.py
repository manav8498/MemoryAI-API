#!/usr/bin/env python3.11
"""
Comprehensive Load Test for Memory AI API
Tests all critical endpoints with realistic user behavior
"""
from locust import HttpUser, task, between, events
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3NjE4NjMzMTgsImlhdCI6MTc2MTg1OTcxOCwidHlwZSI6ImFjY2VzcyJ9.Awx2lMclUgDRAXGpwSRd5q0R9NzU4EFDPHOcKqbzfAA"
COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"

# Realistic search queries for testing
SEARCH_QUERIES = [
    "CVS Health",
    "job applications",
    "companies I applied to",
    "Stripe",
    "Slack",
    "Figma",
    "Notion",
    "when did I apply",
    "product designer",
    "software engineer",
    "internship",
    "full-time",
    "remote jobs",
    "applied on",
    "interview",
]

class MemoryAIUser(HttpUser):
    """Simulates a typical user interacting with Memory AI API"""

    # Wait between 1-3 seconds between tasks (realistic user behavior)
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts"""
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    @task(10)  # Most common operation - weight of 10
    def search_memories(self):
        """Test the search endpoint - most critical operation"""
        query = random.choice(SEARCH_QUERIES)

        with self.client.post(
            "/v1/search",
            json={
                "query": query,
                "collection_id": COLLECTION_ID,
                "top_k": 5
            },
            headers=self.headers,
            catch_response=True,
            name="/v1/search"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    response.success()
                else:
                    response.failure(f"No results field in response: {data}")
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)  # Medium frequency - weight of 5
    def search_with_filters(self):
        """Test search with different parameters"""
        query = random.choice(SEARCH_QUERIES)
        top_k = random.choice([3, 5, 10])

        with self.client.post(
            "/v1/search",
            json={
                "query": query,
                "collection_id": COLLECTION_ID,
                "top_k": top_k
            },
            headers=self.headers,
            catch_response=True,
            name="/v1/search [filtered]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)  # Less frequent - weight of 2
    def health_check(self):
        """Test health endpoint"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data}")
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)  # Rare operation - weight of 1
    def get_collections(self):
        """Test collections endpoint"""
        with self.client.get(
            "/v1/collections",
            headers=self.headers,
            catch_response=True,
            name="/v1/collections"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class IntensiveSearchUser(HttpUser):
    """Simulates power users doing intensive searches"""

    # Faster pace - 0.5 to 1.5 seconds between tasks
    wait_time = between(0.5, 1.5)

    def on_start(self):
        """Called when a user starts"""
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    @task(1)
    def rapid_searches(self):
        """Simulate rapid consecutive searches"""
        for _ in range(3):
            query = random.choice(SEARCH_QUERIES)
            self.client.post(
                "/v1/search",
                json={
                    "query": query,
                    "collection_id": COLLECTION_ID,
                    "top_k": 10
                },
                headers=self.headers,
                name="/v1/search [rapid]"
            )


# Event listeners for detailed reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("\n" + "="*80)
    print("🚀 MEMORY AI LOAD TEST STARTING")
    print("="*80)
    print(f"Target: {environment.host}")
    print(f"Test scenario: Realistic user behavior simulation")
    print("="*80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    stats = environment.stats
    print("\n" + "="*80)
    print("✅ LOAD TEST COMPLETED")
    print("="*80)
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Failure Rate: {stats.total.fail_ratio * 100:.2f}%")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {stats.total.min_response_time:.2f}ms")
    print(f"Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")
    print("="*80 + "\n")
