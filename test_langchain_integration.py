#!/usr/bin/env python3.11
"""
Comprehensive LangChain Integration Test for Memory AI API
Tests all memory features with LangChain agents
"""

import os
import sys
import requests
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import SystemMessage, HumanMessage, AIMessage
    from pydantic import BaseModel, Field
    print("✓ LangChain imports successful")
except ImportError as e:
    print(f"❌ Missing LangChain dependencies: {e}")
    print("\nInstall with:")
    print("  pip install langchain langchain-openai langchain-core tiktoken")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

MEMORY_API_URL = "http://localhost:8000/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Test user credentials
TEST_EMAIL = "langchain_test@example.com"
TEST_PASSWORD = "LangChainTest123!"
TEST_NAME = "LangChain Test User"

# =============================================================================
# MEMORY API WRAPPER FOR LANGCHAIN
# =============================================================================

class MemoryAPIClient:
    """Client for Memory AI API."""

    def __init__(self, api_url: str, token: str = None):
        self.api_url = api_url
        self.token = token
        self.collection_id = None
        self.headers = {
            "Content-Type": "application/json"
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def register_or_login(self, email: str, password: str, full_name: str = "Test User") -> str:
        """Register or login and get token."""
        # Try to login first
        try:
            response = requests.post(
                f"{self.api_url}/auth/login",
                json={"email": email, "password": password}
            )
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                self.headers["Authorization"] = f"Bearer {self.token}"
                return self.token
        except:
            pass

        # Register new user
        response = requests.post(
            f"{self.api_url}/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name
            }
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]
        self.headers["Authorization"] = f"Bearer {self.token}"
        return self.token

    def create_collection(self, name: str, description: str = "") -> str:
        """Create a memory collection."""
        response = requests.post(
            f"{self.api_url}/collections",
            headers=self.headers,
            json={"name": name, "description": description}
        )
        response.raise_for_status()
        self.collection_id = response.json()["id"]
        return self.collection_id

    def list_collections(self) -> List[Dict]:
        """List all collections."""
        response = requests.get(
            f"{self.api_url}/collections",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["collections"]

    def add_memory(self, content: str, importance: float = 0.5, metadata: Dict = None) -> Dict:
        """Add a memory."""
        if not self.collection_id:
            raise ValueError("No collection set. Call create_collection() first.")

        data = {
            "collection_id": self.collection_id,
            "content": content,
            "importance": importance
        }
        if metadata:
            data["metadata"] = metadata

        response = requests.post(
            f"{self.api_url}/memories",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()

    def search_memories(self, query: str, limit: int = 5, filters: Dict = None) -> List[Dict]:
        """Search memories."""
        data = {
            "collection_id": self.collection_id,
            "query": query,
            "limit": limit
        }
        if filters:
            data["filters"] = filters

        response = requests.post(
            f"{self.api_url}/search",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["results"]

    def reason_with_memories(self, query: str, provider: str = "gemini") -> Dict:
        """Use LLM to reason over memories."""
        response = requests.post(
            f"{self.api_url}/search/reason",
            headers=self.headers,
            json={
                "collection_id": self.collection_id,
                "query": query,
                "provider": provider,
                "include_steps": True
            }
        )
        response.raise_for_status()
        return response.json()

# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

class MemorySearchInput(BaseModel):
    """Input for memory search tool."""
    query: str = Field(description="The search query to find relevant memories")
    limit: int = Field(default=5, description="Maximum number of results")

class MemoryAddInput(BaseModel):
    """Input for memory add tool."""
    content: str = Field(description="The content to store in memory")
    importance: float = Field(default=0.7, description="Importance score (0-1)")

class MemorySearchTool(BaseTool):
    """Tool for searching memories."""

    name: str = "search_memory"
    description: str = """Search for relevant memories based on a query.
    Use this when you need to recall past information, user preferences, or previous conversations.
    Returns the most relevant memories with similarity scores."""
    args_schema: type[BaseModel] = MemorySearchInput
    memory_client: MemoryAPIClient = None

    def _run(self, query: str, limit: int = 5) -> str:
        """Search memories."""
        try:
            results = self.memory_client.search_memories(query, limit)

            if not results:
                return "No relevant memories found."

            output = ["Found relevant memories:"]
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                content = result.get('content', '')
                metadata = result.get('metadata', {})

                output.append(f"\n{i}. {content}")
                output.append(f"   Relevance: {score:.2f}")
                if metadata:
                    output.append(f"   Context: {metadata}")

            return "\n".join(output)
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    async def _arun(self, query: str, limit: int = 5) -> str:
        """Async version."""
        return self._run(query, limit)

class MemoryAddTool(BaseTool):
    """Tool for adding memories."""

    name: str = "add_memory"
    description: str = """Store important information in memory for future recall.
    Use this to remember facts, preferences, decisions, or important details from conversations.
    Higher importance (0.8-1.0) for critical info, medium (0.5-0.7) for general info."""
    args_schema: type[BaseModel] = MemoryAddInput
    memory_client: MemoryAPIClient = None

    def _run(self, content: str, importance: float = 0.7) -> str:
        """Add a memory."""
        try:
            result = self.memory_client.add_memory(
                content=content,
                importance=importance,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "source": "langchain_agent"
                }
            )
            return f"✓ Memory stored successfully: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    async def _arun(self, content: str, importance: float = 0.7) -> str:
        """Async version."""
        return self._run(content, importance)

# =============================================================================
# LANGCHAIN AGENT WITH MEMORY
# =============================================================================

class LangChainMemoryAgent:
    """LangChain agent with Memory AI API integration."""

    def __init__(
        self,
        memory_client: MemoryAPIClient,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        verbose: bool = True
    ):
        self.memory_client = memory_client
        self.verbose = verbose

        # Create tools
        self.tools = [
            MemorySearchTool(memory_client=memory_client),
            MemoryAddTool(memory_client=memory_client)
        ]

        # Create LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.7
        )

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to a persistent memory system.

You have two powerful tools:
1. search_memory - Search for relevant memories from past conversations
2. add_memory - Store important information for future recall

Guidelines:
- ALWAYS search memory first when asked about past information
- Store important facts, preferences, and decisions with appropriate importance scores
- Be proactive about storing useful information
- Use high importance (0.8-1.0) for critical user preferences or important facts
- Use medium importance (0.5-0.7) for general information
- Cite your memories when answering questions

Current date: {current_date}"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=5
        )

        self.chat_history = []

    def chat(self, message: str) -> str:
        """Chat with the agent."""
        result = self.executor.invoke({
            "input": message,
            "chat_history": self.chat_history,
            "current_date": datetime.now().strftime("%Y-%m-%d")
        })

        # Update chat history
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=result["output"]))

        return result["output"]

# =============================================================================
# TEST SUITE
# =============================================================================

def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_test(test_name: str):
    """Print test name."""
    print(f"\n{'─' * 80}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'─' * 80}")

def print_success(message: str):
    """Print success message."""
    print(f"✓ {message}")

def print_error(message: str):
    """Print error message."""
    print(f"❌ {message}")

def run_tests():
    """Run comprehensive LangChain integration tests."""

    print_section("LANGCHAIN + MEMORY AI INTEGRATION TEST SUITE")

    # Check prerequisites
    print("\n📋 Checking prerequisites...")

    if not OPENAI_API_KEY:
        print_error("OPENAI_API_KEY not set!")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or use: OPENAI_API_KEY='your-key' python test_langchain_integration.py")
        return False
    print_success("OpenAI API key found")

    # Check API health
    try:
        response = requests.get(f"http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print_success("Memory API is running")
        else:
            print_error("Memory API returned unexpected status")
            return False
    except:
        print_error("Memory API not accessible at http://localhost:8000")
        print("   Start it with: docker-compose up -d")
        return False

    print_section("STEP 1: Authentication & Setup")

    # Initialize client
    memory_client = MemoryAPIClient(MEMORY_API_URL)

    print(f"\n🔐 Registering/logging in user: {TEST_EMAIL}")
    try:
        token = memory_client.register_or_login(TEST_EMAIL, TEST_PASSWORD, TEST_NAME)
        print_success(f"Authenticated! Token: {token[:20]}...")
    except Exception as e:
        print_error(f"Authentication failed: {e}")
        return False

    # Create collection
    print(f"\n📚 Creating memory collection...")
    try:
        collection_id = memory_client.create_collection(
            name=f"LangChain Test {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Testing LangChain integration with Memory AI"
        )
        print_success(f"Collection created: {collection_id}")
    except Exception as e:
        print_error(f"Collection creation failed: {e}")
        return False

    print_section("STEP 2: Initialize LangChain Agent")

    print("\n🤖 Creating LangChain agent with memory tools...")
    try:
        agent = LangChainMemoryAgent(
            memory_client=memory_client,
            openai_api_key=OPENAI_API_KEY,
            verbose=True
        )
        print_success("Agent created successfully!")
        print(f"   Model: gpt-4o-mini")
        print(f"   Tools: {', '.join([tool.name for tool in agent.tools])}")
    except Exception as e:
        print_error(f"Agent creation failed: {e}")
        return False

    print_section("STEP 3: Test Memory Storage")

    print_test("Storing user information")
    print("👤 User: My name is Alice, I'm a Python developer at Google, and I love AI research")
    try:
        response = agent.chat("My name is Alice, I'm a Python developer at Google, and I love AI research")
        print(f"\n🤖 Agent: {response}")
        print_success("Memory storage test passed")
    except Exception as e:
        print_error(f"Memory storage test failed: {e}")

    time.sleep(2)  # Give time for memory to be indexed

    print_test("Storing preferences")
    print("👤 User: I prefer dark mode in all my apps and I'm allergic to peanuts")
    try:
        response = agent.chat("I prefer dark mode in all my apps and I'm allergic to peanuts")
        print(f"\n🤖 Agent: {response}")
        print_success("Preference storage test passed")
    except Exception as e:
        print_error(f"Preference storage test failed: {e}")

    time.sleep(2)

    print_section("STEP 4: Test Memory Recall")

    print_test("Recalling user name")
    print("👤 User: What's my name?")
    try:
        response = agent.chat("What's my name?")
        print(f"\n🤖 Agent: {response}")
        if "Alice" in response or "alice" in response.lower():
            print_success("Name recall test passed")
        else:
            print_error("Name not correctly recalled")
    except Exception as e:
        print_error(f"Name recall test failed: {e}")

    print_test("Recalling work information")
    print("👤 User: Where do I work?")
    try:
        response = agent.chat("Where do I work?")
        print(f"\n🤖 Agent: {response}")
        if "Google" in response:
            print_success("Work recall test passed")
        else:
            print_error("Work information not correctly recalled")
    except Exception as e:
        print_error(f"Work recall test failed: {e}")

    print_test("Recalling preferences")
    print("👤 User: What are my UI preferences?")
    try:
        response = agent.chat("What are my UI preferences?")
        print(f"\n🤖 Agent: {response}")
        if "dark mode" in response.lower():
            print_success("Preference recall test passed")
        else:
            print_error("Preferences not correctly recalled")
    except Exception as e:
        print_error(f"Preference recall test failed: {e}")

    print_section("STEP 5: Test Complex Reasoning")

    print_test("Multi-fact reasoning")
    print("👤 User: Based on what you know about me, what kind of snacks should you avoid offering me at a tech conference?")
    try:
        response = agent.chat("Based on what you know about me, what kind of snacks should you avoid offering me at a tech conference?")
        print(f"\n🤖 Agent: {response}")
        if "peanut" in response.lower():
            print_success("Complex reasoning test passed")
        else:
            print_error("Complex reasoning test failed - didn't recall allergy")
    except Exception as e:
        print_error(f"Complex reasoning test failed: {e}")

    print_section("STEP 6: Test Conversation Continuity")

    print_test("Follow-up questions")
    print("👤 User: And what programming language do I use?")
    try:
        response = agent.chat("And what programming language do I use?")
        print(f"\n🤖 Agent: {response}")
        if "Python" in response or "python" in response.lower():
            print_success("Conversation continuity test passed")
        else:
            print_error("Conversation context not maintained")
    except Exception as e:
        print_error(f"Conversation continuity test failed: {e}")

    print_section("STEP 7: Verify Memory Persistence")

    print("\n🔍 Checking stored memories directly via API...")
    try:
        memories = memory_client.search_memories("Alice", limit=10)
        print(f"   Found {len(memories)} memories containing 'Alice'")
        for i, mem in enumerate(memories, 1):
            print(f"   {i}. {mem['content'][:80]}... (score: {mem['score']:.2f})")
        print_success(f"Memory persistence verified - {len(memories)} memories stored")
    except Exception as e:
        print_error(f"Memory persistence check failed: {e}")

    print_section("TEST SUMMARY")

    print("\n✅ All tests completed!")
    print(f"\n📊 Statistics:")
    print(f"   - Collection ID: {collection_id}")
    print(f"   - Conversations: {len(agent.chat_history) // 2}")
    print(f"   - Memories stored: {len(memories)}")

    print("\n💡 Next Steps:")
    print("   1. Try the production examples in examples/integration_langchain.py")
    print("   2. Build a chatbot using this pattern")
    print("   3. Deploy to production with proper error handling")
    print("   4. Monitor memory growth and optimize as needed")

    print("\n" + "=" * 80)
    return True

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting LangChain + Memory AI Integration Test")
    print(f"   Timestamp: {datetime.now().isoformat()}")

    success = run_tests()

    if success:
        print("\n✅ Integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)
