#!/usr/bin/env python3
"""
LangChain Integration with Memory API
Demonstrates how to use Memory API with LangChain agents
"""

import requests
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

# =============================================================================
# MEMORY API WRAPPER FOR LANGCHAIN
# =============================================================================

class MemoryAPIWrapper:
    """Wrapper for Memory API to use with LangChain."""
    
    def __init__(self, api_url: str, token: str, collection_id: str):
        self.api_url = api_url
        self.token = token
        self.collection_id = collection_id
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def add_memory(self, content: str, importance: float = 0.5) -> Dict:
        """Add a memory."""
        response = requests.post(
            f"{self.api_url}/memories",
            headers=self.headers,
            json={
                "collection_id": self.collection_id,
                "content": content,
                "importance": importance
            }
        )
        response.raise_for_status()
        return response.json()
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memories."""
        response = requests.post(
            f"{self.api_url}/search",
            headers=self.headers,
            json={
                "collection_id": self.collection_id,
                "query": query,
                "limit": limit
            }
        )
        response.raise_for_status()
        return response.json()["results"]


# =============================================================================
# LANGCHAIN TOOLS FOR MEMORY API
# =============================================================================

class MemorySearchInput(BaseModel):
    """Input for memory search tool."""
    query: str = Field(description="The search query to find relevant memories")
    limit: int = Field(default=5, description="Maximum number of results to return")


class MemoryAddInput(BaseModel):
    """Input for memory add tool."""
    content: str = Field(description="The content to store in memory")
    importance: float = Field(default=0.5, description="Importance score (0-1)")


class MemorySearchTool(BaseTool):
    """Tool for searching memories."""
    
    name: str = "search_memory"
    description: str = "Search for relevant memories based on a query. Use this to recall past information."
    args_schema: type[BaseModel] = MemorySearchInput
    memory_api: MemoryAPIWrapper = None
    
    def _run(self, query: str, limit: int = 5) -> str:
        """Search memories."""
        results = self.memory_api.search(query, limit)
        
        if not results:
            return "No relevant memories found."
        
        output = ["Found relevant memories:"]
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['content']} (relevance: {result['score']:.2f})")
        
        return "\n".join(output)
    
    async def _arun(self, query: str, limit: int = 5) -> str:
        """Async version."""
        return self._run(query, limit)


class MemoryAddTool(BaseTool):
    """Tool for adding memories."""
    
    name: str = "add_memory"
    description: str = "Store important information in memory for future recall. Use this to remember facts, preferences, or important details."
    args_schema: type[BaseModel] = MemoryAddInput
    memory_api: MemoryAPIWrapper = None
    
    def _run(self, content: str, importance: float = 0.5) -> str:
        """Add a memory."""
        result = self.memory_api.add_memory(content, importance)
        return f"Memory stored successfully: {content}"
    
    async def _arun(self, content: str, importance: float = 0.5) -> str:
        """Async version."""
        return self._run(content, importance)


# =============================================================================
# LANGCHAIN AGENT WITH MEMORY
# =============================================================================

class LangChainMemoryAgent:
    """LangChain agent with Memory API integration."""
    
    def __init__(
        self,
        memory_api: MemoryAPIWrapper,
        openai_api_key: str,
        model: str = "gpt-4"
    ):
        self.memory_api = memory_api
        
        # Create tools
        self.tools = [
            MemorySearchTool(memory_api=memory_api),
            MemoryAddTool(memory_api=memory_api)
        ]
        
        # Create LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.7
        )
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant with access to a memory system.

You can:
1. Search your memory to recall past information
2. Store new information in memory for future use

Always search your memory first before answering questions about past interactions or stored information.
When you learn something important, store it in memory."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        self.chat_history = []
    
    def chat(self, message: str) -> str:
        """Chat with the agent."""
        result = self.executor.invoke({
            "input": message,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=result["output"]))
        
        return result["output"]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Demonstrate LangChain integration."""
    
    print("=" * 70)
    print("🦜 LANGCHAIN AGENT WITH MEMORY API")
    print("=" * 70)
    
    # Configuration
    MEMORY_API_URL = "http://localhost:8000/v1"
    EMAIL = "your_email@example.com"
    PASSWORD = "your_password"
    OPENAI_API_KEY = "your_openai_api_key"
    
    # Step 1: Login to Memory API
    print("\n1️⃣ Authenticating with Memory API...")
    response = requests.post(
        f"{MEMORY_API_URL}/auth/login",
        json={"email": EMAIL, "password": PASSWORD}
    )
    token = response.json()["access_token"]
    print("✓ Authenticated!")
    
    # Step 2: Create collection
    print("\n2️⃣ Creating memory collection...")
    response = requests.post(
        f"{MEMORY_API_URL}/collections",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "name": "LangChain Agent Memory",
            "description": "Memory for LangChain agent demo"
        }
    )
    collection_id = response.json()["id"]
    print(f"✓ Collection created: {collection_id}")
    
    # Step 3: Initialize Memory API wrapper
    print("\n3️⃣ Initializing Memory API wrapper...")
    memory_api = MemoryAPIWrapper(MEMORY_API_URL, token, collection_id)
    print("✓ Memory API ready!")
    
    # Step 4: Create LangChain agent
    print("\n4️⃣ Creating LangChain agent...")
    agent = LangChainMemoryAgent(
        memory_api=memory_api,
        openai_api_key=OPENAI_API_KEY
    )
    print("✓ Agent created!")
    
    # Step 5: Interact with agent
    print("\n5️⃣ Starting conversation...")
    print("=" * 70)
    
    # The agent will automatically use memory tools
    print("\n👤 User: My name is Bob and I love Python programming")
    response = agent.chat("My name is Bob and I love Python programming")
    print(f"🤖 Agent: {response}")
    
    print("\n👤 User: What's my name?")
    response = agent.chat("What's my name?")
    print(f"🤖 Agent: {response}")
    
    print("\n👤 User: What do I love?")
    response = agent.chat("What do I love?")
    print(f"🤖 Agent: {response}")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

