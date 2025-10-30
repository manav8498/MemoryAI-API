#!/usr/bin/env python3
"""
Simple AI Agent with Memory Integration
Demonstrates how to integrate ANY AI agent with the Memory API
"""

import requests
import json
from typing import List, Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

MEMORY_API_URL = "http://localhost:8000/v1"
EMAIL = "your_email@example.com"
PASSWORD = "your_password"

# =============================================================================
# STEP 1: AUTHENTICATION
# =============================================================================

def login() -> str:
    """Login and get access token."""
    response = requests.post(
        f"{MEMORY_API_URL}/auth/login",
        json={"email": EMAIL, "password": PASSWORD}
    )
    response.raise_for_status()
    return response.json()["access_token"]


def register(email: str, password: str, full_name: str) -> str:
    """Register new user and get access token."""
    response = requests.post(
        f"{MEMORY_API_URL}/auth/register",
        json={
            "email": email,
            "password": password,
            "full_name": full_name
        }
    )
    response.raise_for_status()
    return response.json()["access_token"]


# =============================================================================
# STEP 2: MEMORY OPERATIONS
# =============================================================================

class MemoryAPI:
    """Simple wrapper for Memory API operations."""
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.collection_id = None
    
    def create_collection(self, name: str, description: str = "") -> str:
        """Create a memory collection."""
        response = requests.post(
            f"{MEMORY_API_URL}/collections",
            headers=self.headers,
            json={"name": name, "description": description}
        )
        response.raise_for_status()
        self.collection_id = response.json()["id"]
        return self.collection_id
    
    def add_memory(self, content: str, importance: float = 0.5, metadata: Dict = None) -> Dict:
        """Add a memory to the collection."""
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
            f"{MEMORY_API_URL}/memories",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant memories."""
        response = requests.post(
            f"{MEMORY_API_URL}/search",
            headers=self.headers,
            json={
                "collection_id": self.collection_id,
                "query": query,
                "limit": limit
            }
        )
        response.raise_for_status()
        return response.json()["results"]
    
    def reason_with_memories(self, query: str, provider: str = "gemini") -> Dict:
        """Use LLM to reason over memories."""
        response = requests.post(
            f"{MEMORY_API_URL}/search/reason",
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
# STEP 3: SIMPLE AI AGENT WITH MEMORY
# =============================================================================

class SimpleAIAgent:
    """
    A simple AI agent that uses the Memory API.
    
    This demonstrates the basic pattern:
    1. User asks a question
    2. Agent searches memories for context
    3. Agent generates response using context
    4. Agent stores the interaction as a new memory
    """
    
    def __init__(self, memory_api: MemoryAPI, name: str = "Assistant"):
        self.memory = memory_api
        self.name = name
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """
        Process user message with memory context.
        
        Args:
            user_message: The user's input
            
        Returns:
            Agent's response
        """
        print(f"\n🧠 Searching memories for: '{user_message}'...")
        
        # Search for relevant memories
        relevant_memories = self.memory.search_memories(
            query=user_message,
            limit=3
        )
        
        # Build context from memories
        context = self._build_context(relevant_memories)
        
        # Generate response using LLM reasoning
        print(f"💭 Reasoning with {len(relevant_memories)} relevant memories...")
        result = self.memory.reason_with_memories(
            query=user_message,
            provider="gemini"
        )
        
        response = result["answer"]
        
        # Store this interaction as a memory
        print(f"💾 Storing interaction as memory...")
        self.memory.add_memory(
            content=f"User asked: {user_message}. Assistant answered: {response}",
            importance=0.6,
            metadata={
                "type": "conversation",
                "user_message": user_message,
                "agent_response": response
            }
        )
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_message,
            "assistant": response,
            "memories_used": len(relevant_memories)
        })
        
        return response
    
    def _build_context(self, memories: List[Dict]) -> str:
        """Build context string from memories."""
        if not memories:
            return "No relevant memories found."
        
        context_parts = ["Relevant memories:"]
        for i, mem in enumerate(memories, 1):
            context_parts.append(
                f"{i}. {mem['content']} (relevance: {mem['score']:.2f})"
            )
        return "\n".join(context_parts)
    
    def learn_fact(self, fact: str, importance: float = 0.7):
        """Explicitly teach the agent a new fact."""
        print(f"📚 Learning: {fact}")
        self.memory.add_memory(
            content=fact,
            importance=importance,
            metadata={"type": "learned_fact"}
        )
        print("✓ Fact stored in memory!")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Demonstrate AI agent with memory integration."""
    
    print("=" * 70)
    print("🤖 SIMPLE AI AGENT WITH MEMORY INTEGRATION")
    print("=" * 70)
    
    # Step 1: Authenticate
    print("\n1️⃣ Authenticating...")
    try:
        token = login()
        print("✓ Logged in successfully!")
    except:
        print("⚠️ Login failed. Make sure you have an account or update credentials.")
        return
    
    # Step 2: Initialize Memory API
    print("\n2️⃣ Initializing Memory API...")
    memory = MemoryAPI(token)
    collection_id = memory.create_collection(
        name="AI Agent Memory",
        description="Memory for simple AI agent demo"
    )
    print(f"✓ Collection created: {collection_id}")
    
    # Step 3: Create AI Agent
    print("\n3️⃣ Creating AI Agent...")
    agent = SimpleAIAgent(memory, name="MemoryBot")
    print("✓ Agent ready!")
    
    # Step 4: Teach the agent some facts
    print("\n4️⃣ Teaching agent some facts...")
    agent.learn_fact("The user's name is Alice", importance=0.9)
    agent.learn_fact("Alice prefers Python over JavaScript", importance=0.7)
    agent.learn_fact("Alice is working on a machine learning project", importance=0.8)
    agent.learn_fact("Alice's favorite color is blue", importance=0.5)
    
    # Step 5: Have a conversation
    print("\n5️⃣ Starting conversation...")
    print("=" * 70)
    
    # Question 1
    print("\n👤 User: What's my name?")
    response = agent.chat("What's my name?")
    print(f"🤖 {agent.name}: {response}")
    
    # Question 2
    print("\n👤 User: What programming language do I prefer?")
    response = agent.chat("What programming language do I prefer?")
    print(f"🤖 {agent.name}: {response}")
    
    # Question 3
    print("\n👤 User: What am I working on?")
    response = agent.chat("What am I working on?")
    print(f"🤖 {agent.name}: {response}")
    
    # Question 4 - Tests memory of conversation
    print("\n👤 User: What did we just talk about?")
    response = agent.chat("What did we just talk about?")
    print(f"🤖 {agent.name}: {response}")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print(f"📊 Conversation history: {len(agent.conversation_history)} exchanges")
    print("=" * 70)


if __name__ == "__main__":
    main()

