#!/usr/bin/env python3
"""
AutoGen Integration with Memory API
Demonstrates how to use Memory API with Microsoft AutoGen agents
"""

import requests
from typing import List, Dict, Any, Optional
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# =============================================================================
# MEMORY API CLIENT
# =============================================================================

class MemoryAPIClient:
    """Simple client for Memory API."""
    
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
    
    def reason(self, query: str, provider: str = "gemini") -> str:
        """Reason over memories."""
        response = requests.post(
            f"{self.api_url}/search/reason",
            headers=self.headers,
            json={
                "collection_id": self.collection_id,
                "query": query,
                "provider": provider
            }
        )
        response.raise_for_status()
        return response.json()["answer"]


# =============================================================================
# AUTOGEN AGENT WITH MEMORY
# =============================================================================

class MemoryEnhancedAgent:
    """AutoGen agent enhanced with Memory API."""
    
    def __init__(
        self,
        name: str,
        memory_client: MemoryAPIClient,
        llm_config: Dict,
        system_message: str = None
    ):
        self.memory = memory_client
        
        # Default system message with memory instructions
        if system_message is None:
            system_message = f"""You are {name}, an AI assistant with access to a memory system.

Before answering questions, you can search your memory for relevant information.
After conversations, you should store important information in memory.

To search memory, say: SEARCH_MEMORY: <query>
To store memory, say: STORE_MEMORY: <content>

Always provide helpful and accurate responses based on your memory."""
        
        # Create AutoGen assistant
        self.agent = AssistantAgent(
            name=name,
            llm_config=llm_config,
            system_message=system_message
        )
        
        # Register memory functions
        self._register_memory_functions()
    
    def _register_memory_functions(self):
        """Register memory-related functions."""
        
        @self.agent.register_for_execution()
        @self.agent.register_for_llm(description="Search memory for relevant information")
        def search_memory(query: str) -> str:
            """Search memory for relevant information."""
            results = self.memory.search(query, limit=5)
            if not results:
                return "No relevant memories found."
            
            output = ["Relevant memories:"]
            for i, result in enumerate(results, 1):
                output.append(f"{i}. {result['content']} (score: {result['score']:.2f})")
            return "\n".join(output)
        
        @self.agent.register_for_execution()
        @self.agent.register_for_llm(description="Store important information in memory")
        def store_memory(content: str, importance: float = 0.7) -> str:
            """Store important information in memory."""
            self.memory.add_memory(content, importance)
            return f"Stored in memory: {content}"
    
    def get_agent(self) -> AssistantAgent:
        """Get the underlying AutoGen agent."""
        return self.agent


# =============================================================================
# MULTI-AGENT SYSTEM WITH SHARED MEMORY
# =============================================================================

class MultiAgentMemorySystem:
    """Multi-agent system with shared memory."""
    
    def __init__(
        self,
        memory_client: MemoryAPIClient,
        llm_config: Dict
    ):
        self.memory = memory_client
        self.llm_config = llm_config
        
        # Create specialized agents
        self.researcher = MemoryEnhancedAgent(
            name="Researcher",
            memory_client=memory_client,
            llm_config=llm_config,
            system_message="""You are a Researcher. Your job is to:
1. Search memory for relevant information
2. Analyze and synthesize information
3. Store new findings in memory
Always search memory before providing answers."""
        )
        
        self.writer = MemoryEnhancedAgent(
            name="Writer",
            memory_client=memory_client,
            llm_config=llm_config,
            system_message="""You are a Writer. Your job is to:
1. Take information from the Researcher
2. Create well-written, clear responses
3. Store important writing patterns in memory"""
        )
        
        # Create user proxy
        self.user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
        
        # Create group chat
        self.group_chat = GroupChat(
            agents=[
                self.user_proxy,
                self.researcher.get_agent(),
                self.writer.get_agent()
            ],
            messages=[],
            max_round=10
        )
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config
        )
    
    def run(self, task: str) -> str:
        """Run the multi-agent system on a task."""
        self.user_proxy.initiate_chat(
            self.manager,
            message=task
        )
        
        # Get the final response
        return self.group_chat.messages[-1]["content"]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Demonstrate AutoGen integration."""
    
    print("=" * 70)
    print("🤖 AUTOGEN AGENTS WITH MEMORY API")
    print("=" * 70)
    
    # Configuration
    MEMORY_API_URL = "http://localhost:8000/v1"
    EMAIL = "your_email@example.com"
    PASSWORD = "your_password"
    
    # LLM configuration for AutoGen
    llm_config = {
        "model": "gpt-4",
        "api_key": "your_openai_api_key",
        "temperature": 0.7
    }
    
    # Step 1: Setup Memory API
    print("\n1️⃣ Setting up Memory API...")
    
    # Login
    response = requests.post(
        f"{MEMORY_API_URL}/auth/login",
        json={"email": EMAIL, "password": PASSWORD}
    )
    token = response.json()["access_token"]
    
    # Create collection
    response = requests.post(
        f"{MEMORY_API_URL}/collections",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "name": "AutoGen Multi-Agent Memory",
            "description": "Shared memory for AutoGen agents"
        }
    )
    collection_id = response.json()["id"]
    
    memory_client = MemoryAPIClient(MEMORY_API_URL, token, collection_id)
    print("✓ Memory API ready!")
    
    # Step 2: Pre-populate some knowledge
    print("\n2️⃣ Pre-populating knowledge base...")
    memory_client.add_memory(
        "Python is a high-level programming language known for readability",
        importance=0.8
    )
    memory_client.add_memory(
        "Machine learning is a subset of AI that learns from data",
        importance=0.9
    )
    memory_client.add_memory(
        "AutoGen is a framework for building multi-agent systems",
        importance=0.85
    )
    print("✓ Knowledge base populated!")
    
    # Step 3: Create single agent example
    print("\n3️⃣ Single Agent Example...")
    print("-" * 70)
    
    single_agent = MemoryEnhancedAgent(
        name="MemoryBot",
        memory_client=memory_client,
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False
    )
    
    # Ask a question
    print("\n👤 User: What do you know about Python?")
    user_proxy.initiate_chat(
        single_agent.get_agent(),
        message="What do you know about Python? Search your memory first."
    )
    
    # Step 4: Multi-agent example
    print("\n4️⃣ Multi-Agent System Example...")
    print("-" * 70)
    
    multi_agent_system = MultiAgentMemorySystem(
        memory_client=memory_client,
        llm_config=llm_config
    )
    
    print("\n👤 User: Research and write about machine learning")
    result = multi_agent_system.run(
        "Research what we know about machine learning and write a brief explanation."
    )
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

