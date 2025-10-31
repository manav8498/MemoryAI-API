#!/usr/bin/env python3.11
"""
Quick Start: LangChain + Memory AI in 5 Minutes
A minimal example to get you started quickly
"""

import os
import requests
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# =============================================================================
# CONFIGURATION - CHANGE THESE!
# =============================================================================

MEMORY_API_URL = "http://localhost:8000/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")

# User credentials (will create if doesn't exist)
EMAIL = "test@example.com"
PASSWORD = "TestPassword123!"

# =============================================================================
# SIMPLE MEMORY CLIENT
# =============================================================================

class MemoryClient:
    """Minimal Memory API client."""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.token = None
        self.collection_id = None

    def login(self, email: str, password: str):
        """Login and get token."""
        # Try login
        try:
            r = requests.post(f"{self.api_url}/auth/login",
                            json={"email": email, "password": password})
            self.token = r.json()["access_token"]
            return
        except:
            pass

        # Register if login fails
        r = requests.post(f"{self.api_url}/auth/register",
                         json={"email": email, "password": password, "full_name": "Test User"})
        self.token = r.json()["access_token"]

    def setup_collection(self, name: str):
        """Create collection."""
        r = requests.post(
            f"{self.api_url}/collections",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json={"name": name, "description": "LangChain memories"}
        )
        self.collection_id = r.json()["id"]

    def add_memory(self, content: str, importance: float = 0.7):
        """Store a memory."""
        requests.post(
            f"{self.api_url}/memories",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json={
                "collection_id": self.collection_id,
                "content": content,
                "importance": importance
            }
        )

    def search(self, query: str, limit: int = 5):
        """Search memories."""
        r = requests.post(
            f"{self.api_url}/search",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json={
                "collection_id": self.collection_id,
                "query": query,
                "limit": limit
            }
        )
        return r.json()["results"]

# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

class AddInput(BaseModel):
    content: str = Field(description="Content to store")
    importance: float = Field(default=0.7, description="Importance (0-1)")

class SearchTool(BaseTool):
    name: str = "search_memory"
    description: str = "Search past memories and conversations"
    args_schema: type[BaseModel] = SearchInput
    memory: MemoryClient = None

    def _run(self, query: str) -> str:
        results = self.memory.search(query)
        if not results:
            return "No memories found."
        return "\n".join([f"- {r['content']}" for r in results])

class AddTool(BaseTool):
    name: str = "add_memory"
    description: str = "Store important information for later"
    args_schema: type[BaseModel] = AddInput
    memory: MemoryClient = None

    def _run(self, content: str, importance: float = 0.7) -> str:
        self.memory.add_memory(content, importance)
        return f"Stored: {content}"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n🚀 LangChain + Memory AI Quick Start\n")

    # Step 1: Setup Memory API
    print("1️⃣ Setting up Memory API...")
    memory = MemoryClient(MEMORY_API_URL)
    memory.login(EMAIL, PASSWORD)
    memory.setup_collection("LangChain Demo")
    print("   ✓ Memory API ready\n")

    # Step 2: Create LangChain agent
    print("2️⃣ Creating LangChain agent...")

    tools = [
        SearchTool(memory=memory),
        AddTool(memory=memory)
    ]

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with memory. Use search_memory to recall past info and add_memory to store important facts."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("   ✓ Agent created\n")

    # Step 3: Chat!
    print("3️⃣ Testing conversation with memory...\n")
    print("=" * 60)

    # First message - store info
    print("\n👤 User: My name is Bob and I love Python")
    result = executor.invoke({"input": "My name is Bob and I love Python", "chat_history": []})
    print(f"🤖 Agent: {result['output']}\n")

    # Second message - recall info
    print("👤 User: What's my name?")
    result = executor.invoke({"input": "What's my name?", "chat_history": []})
    print(f"🤖 Agent: {result['output']}\n")

    print("=" * 60)
    print("\n✅ Success! Your agent has persistent memory.\n")
    print("💡 Next steps:")
    print("   - Modify this script for your use case")
    print("   - Run test_langchain_integration.py for comprehensive tests")
    print("   - Check examples/integration_langchain.py for production code\n")

if __name__ == "__main__":
    main()
