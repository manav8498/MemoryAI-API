#!/usr/bin/env python3.11
"""
Production-Ready LangChain Chatbot with Memory AI
A complete example you can deploy to production
"""

import os
import sys
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Application configuration."""
    memory_api_url: str = os.getenv("MEMORY_API_URL", "http://localhost:8000/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    memory_api_email: str = os.getenv("MEMORY_API_EMAIL", "")
    memory_api_password: str = os.getenv("MEMORY_API_PASSWORD", "")
    collection_name: str = os.getenv("COLLECTION_NAME", "ChatBot Memory")
    max_iterations: int = 5
    verbose: bool = True

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY not set")
            return False
        if not self.memory_api_email or not self.memory_api_password:
            logger.error("MEMORY_API_EMAIL and MEMORY_API_PASSWORD must be set")
            return False
        return True

# =============================================================================
# MEMORY API CLIENT (PRODUCTION-READY)
# =============================================================================

class MemoryAPIClient:
    """Production-ready Memory API client with error handling."""

    def __init__(self, api_url: str, timeout: int = 30):
        self.api_url = api_url
        self.token = None
        self.collection_id = None
        self.timeout = timeout
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def authenticate(self, email: str, password: str, full_name: str = "ChatBot User") -> bool:
        """Authenticate and get token."""
        try:
            # Try login
            response = self.session.post(
                f"{self.api_url}/auth/login",
                json={"email": email, "password": password},
                timeout=self.timeout
            )

            if response.status_code == 200:
                self.token = response.json()["access_token"]
                logger.info(f"Logged in as {email}")
                return True

        except Exception as e:
            logger.debug(f"Login failed: {e}, attempting registration")

        # Register if login failed
        try:
            response = self.session.post(
                f"{self.api_url}/auth/register",
                json={"email": email, "password": password, "full_name": full_name},
                timeout=self.timeout
            )
            response.raise_for_status()
            self.token = response.json()["access_token"]
            logger.info(f"Registered and logged in as {email}")
            return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def get_or_create_collection(self, name: str, description: str = "") -> Optional[str]:
        """Get existing collection or create new one."""
        try:
            # List collections
            response = self.session.get(
                f"{self.api_url}/collections",
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            collections = response.json()["collections"]

            # Find existing collection
            for collection in collections:
                if collection["name"] == name:
                    self.collection_id = collection["id"]
                    logger.info(f"Using existing collection: {name} ({self.collection_id})")
                    return self.collection_id

            # Create new collection
            response = self.session.post(
                f"{self.api_url}/collections",
                headers=self._get_headers(),
                json={"name": name, "description": description},
                timeout=self.timeout
            )
            response.raise_for_status()
            self.collection_id = response.json()["id"]
            logger.info(f"Created new collection: {name} ({self.collection_id})")
            return self.collection_id

        except Exception as e:
            logger.error(f"Collection setup failed: {e}")
            return None

    def add_memory(self, content: str, importance: float = 0.7, metadata: Dict = None) -> bool:
        """Add a memory with error handling."""
        try:
            data = {
                "collection_id": self.collection_id,
                "content": content,
                "importance": importance,
                "metadata": metadata or {}
            }

            response = self.session.post(
                f"{self.api_url}/memories",
                headers=self._get_headers(),
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.debug(f"Stored memory: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return False

    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memories with error handling."""
        try:
            response = self.session.post(
                f"{self.api_url}/search",
                headers=self._get_headers(),
                json={
                    "collection_id": self.collection_id,
                    "query": query,
                    "limit": limit
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            results = response.json()["results"]
            logger.debug(f"Found {len(results)} memories for query: {query[:30]}...")
            return results

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

class MemorySearchInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=5, description="Max results")

class MemoryAddInput(BaseModel):
    content: str = Field(description="Content to store")
    importance: float = Field(default=0.7, description="Importance (0-1)")

class MemorySearchTool(BaseTool):
    name: str = "search_memory"
    description: str = "Search for relevant memories from past conversations"
    args_schema: type[BaseModel] = MemorySearchInput
    memory_client: MemoryAPIClient = None

    def _run(self, query: str, limit: int = 5) -> str:
        results = self.memory_client.search_memories(query, limit)
        if not results:
            return "No relevant memories found."

        output = ["Relevant memories:"]
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r['content']} (relevance: {r['score']:.2f})")
        return "\n".join(output)

class MemoryAddTool(BaseTool):
    name: str = "add_memory"
    description: str = "Store important information in long-term memory"
    args_schema: type[BaseModel] = MemoryAddInput
    memory_client: MemoryAPIClient = None

    def _run(self, content: str, importance: float = 0.7) -> str:
        success = self.memory_client.add_memory(content, importance)
        if success:
            return f"✓ Stored in memory: {content}"
        return "Failed to store memory"

# =============================================================================
# CHATBOT
# =============================================================================

class MemoryChatBot:
    """Production chatbot with persistent memory."""

    def __init__(self, config: Config):
        self.config = config
        self.memory_client = None
        self.agent_executor = None
        self.chat_history = []

    def initialize(self) -> bool:
        """Initialize the chatbot."""
        logger.info("Initializing Memory ChatBot...")

        # Setup Memory API
        logger.info("Connecting to Memory API...")
        self.memory_client = MemoryAPIClient(self.config.memory_api_url)

        if not self.memory_client.authenticate(
            self.config.memory_api_email,
            self.config.memory_api_password
        ):
            return False

        if not self.memory_client.get_or_create_collection(
            self.config.collection_name,
            "Persistent memory for chatbot conversations"
        ):
            return False

        # Setup LangChain agent
        logger.info("Creating LangChain agent...")
        tools = [
            MemorySearchTool(memory_client=self.memory_client),
            MemoryAddTool(memory_client=self.memory_client)
        ]

        llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model,
            temperature=0.7
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with persistent memory.

You can search your memory for past conversations and store new information.
Be proactive about:
- Remembering user preferences and important facts
- Recalling past conversations when relevant
- Storing useful information for future interactions

Always be helpful, friendly, and remember what users tell you."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.config.verbose,
            handle_parsing_errors=True,
            max_iterations=self.config.max_iterations
        )

        logger.info("✓ ChatBot initialized successfully!")
        return True

    def chat(self, user_message: str) -> str:
        """Process a user message."""
        try:
            result = self.agent_executor.invoke({
                "input": user_message,
                "chat_history": self.chat_history
            })

            response = result["output"]

            # Update chat history
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(AIMessage(content=response))

            # Keep history manageable (last 10 messages)
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

            return response

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "I encountered an error processing your message. Please try again."

    def run_interactive(self):
        """Run interactive chat session."""
        print("\n" + "=" * 60)
        print("  Memory ChatBot - Interactive Mode")
        print("=" * 60)
        print("\nType 'quit' or 'exit' to stop")
        print("Type 'clear' to clear chat history")
        print("Type 'stats' to see memory statistics\n")

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'clear':
                    self.chat_history = []
                    print("✓ Chat history cleared")
                    continue

                if user_input.lower() == 'stats':
                    print(f"\n📊 Statistics:")
                    print(f"   Chat history: {len(self.chat_history) // 2} exchanges")
                    print(f"   Collection ID: {self.memory_client.collection_id}")
                    continue

                # Get response
                response = self.chat(user_input)
                print(f"\n🤖 Bot: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("\n🤖 Memory ChatBot with LangChain")
    print(f"   Timestamp: {datetime.now().isoformat()}\n")

    # Load configuration
    config = Config()

    if not config.validate():
        print("\n❌ Configuration error!")
        print("\nSet required environment variables:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export MEMORY_API_EMAIL='your@email.com'")
        print("  export MEMORY_API_PASSWORD='your-password'")
        print("\nOr pass them when running:")
        print("  OPENAI_API_KEY='...' MEMORY_API_EMAIL='...' python langchain_chatbot_example.py")
        sys.exit(1)

    # Create and initialize chatbot
    chatbot = MemoryChatBot(config)

    if not chatbot.initialize():
        print("\n❌ Failed to initialize chatbot")
        sys.exit(1)

    # Run interactive session
    chatbot.run_interactive()

if __name__ == "__main__":
    main()
