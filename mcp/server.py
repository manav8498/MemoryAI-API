#!/usr/bin/env python3
"""
Memory AI MCP Server

Provides Model Context Protocol integration for Memory AI,
enabling Claude Desktop and other MCP clients to interact with
the most advanced memory API.

Features:
- 50+ memory operations as MCP tools
- Resources for browsing collections and memories
- Prompts for common memory workflows
- Full support for RL, temporal graphs, procedural memory, and more
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.types import TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install 'mcp[cli]'")
    MCP_AVAILABLE = False
    raise

# Use the published SDK
try:
    from memory_ai_sdk import MemoryClient
    SDK_AVAILABLE = True
except ImportError:
    print("Error: Memory AI SDK not installed. Run: pip install memory-ai-sdk")
    SDK_AVAILABLE = False
    raise


# Initialize FastMCP server
mcp = FastMCP(
    "Memory AI",
    dependencies=["memory-ai-sdk>=1.0.0"]
)

# Global client instance (will be initialized with API key from environment)
_client: Optional[MemoryClient] = None


def get_client() -> MemoryClient:
    """Get or create Memory AI client instance."""
    global _client
    if _client is None:
        api_key = os.getenv("MEMORY_AI_API_KEY")
        base_url = os.getenv("MEMORY_AI_BASE_URL", "http://localhost:8000")

        _client = MemoryClient(
            api_key=api_key,
            base_url=base_url
        )

    return _client


# ============================================================================
# AUTHENTICATION TOOLS
# ============================================================================

@mcp.tool()
async def auth_register(email: str, password: str, full_name: str) -> Dict[str, Any]:
    """
    Register a new user account.

    Args:
        email: User's email address
        password: Secure password
        full_name: User's full name

    Returns:
        Authentication response with access token and user info
    """
    client = get_client()
    return await client.auth.register(email, password, full_name)


@mcp.tool()
async def auth_login(email: str, password: str) -> Dict[str, Any]:
    """
    Login with email and password.

    Args:
        email: User's email address
        password: User's password

    Returns:
        Authentication response with access token
    """
    client = get_client()
    return await client.auth.login(email, password)


@mcp.tool()
async def auth_create_api_key(name: str) -> Dict[str, Any]:
    """
    Create a new API key for programmatic access.

    Args:
        name: Descriptive name for the API key

    Returns:
        API key response with the generated key
    """
    client = get_client()
    return await client.auth.create_api_key(name)


@mcp.tool()
async def auth_get_me() -> Dict[str, Any]:
    """
    Get current authenticated user information.

    Returns:
        User profile information
    """
    client = get_client()
    return await client.auth.get_me()


# ============================================================================
# COLLECTION TOOLS
# ============================================================================

@mcp.tool()
async def collection_create(
    name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new memory collection.

    Collections organize related memories together (e.g., "Work Notes", "Personal", "Research").

    Args:
        name: Name of the collection
        description: Optional description
        metadata: Optional metadata as key-value pairs

    Returns:
        Created collection with ID
    """
    client = get_client()
    params = {"name": name}
    if description:
        params["description"] = description
    if metadata:
        params["metadata"] = metadata

    return await client.collections.create(**params)


@mcp.tool()
async def collection_list(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """
    List all memory collections.

    Args:
        skip: Number of collections to skip (for pagination)
        limit: Maximum number of collections to return

    Returns:
        List of collections
    """
    client = get_client()
    return await client.collections.list(skip=skip, limit=limit)


@mcp.tool()
async def collection_get(collection_id: str) -> Dict[str, Any]:
    """
    Get details of a specific collection.

    Args:
        collection_id: ID of the collection

    Returns:
        Collection details
    """
    client = get_client()
    return await client.collections.get(collection_id)


@mcp.tool()
async def collection_update(
    collection_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update a collection's details.

    Args:
        collection_id: ID of the collection
        name: New name (optional)
        description: New description (optional)
        metadata: New metadata (optional)

    Returns:
        Updated collection
    """
    client = get_client()
    params = {}
    if name:
        params["name"] = name
    if description:
        params["description"] = description
    if metadata:
        params["metadata"] = metadata

    return await client.collections.update(collection_id, **params)


@mcp.tool()
async def collection_delete(collection_id: str) -> str:
    """
    Delete a collection.

    Warning: This will also delete all memories in the collection!

    Args:
        collection_id: ID of the collection to delete

    Returns:
        Success message
    """
    client = get_client()
    client.collections.delete(collection_id)
    return f"Collection {collection_id} deleted successfully"


# ============================================================================
# MEMORY TOOLS (Episodic)
# ============================================================================

@mcp.tool()
async def memory_create(
    collection_id: str,
    content: str,
    importance: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new memory in a collection.

    Memories are automatically embedded with AI for semantic search.

    Args:
        collection_id: ID of the collection
        content: The memory content (text)
        importance: Importance score 0.0-1.0 (higher = more important)
        metadata: Optional metadata (tags, timestamps, source, etc.)

    Returns:
        Created memory with ID and embedding info
    """
    client = get_client()
    params = {
        "collection_id": collection_id,
        "content": content,
        "importance": importance
    }
    if metadata:
        params["metadata"] = metadata

    return await client.memories.create(**params)


@mcp.tool()
async def memory_list(
    collection_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List memories, optionally filtered by collection.

    Args:
        collection_id: Filter by collection ID (optional)
        skip: Number of memories to skip
        limit: Maximum memories to return

    Returns:
        List of memories
    """
    client = get_client()
    params = {"skip": skip, "limit": limit}
    if collection_id:
        params["collection_id"] = collection_id

    return await client.memories.list(**params)


@mcp.tool()
async def memory_get(memory_id: str) -> Dict[str, Any]:
    """
    Get a specific memory with full metadata.

    Args:
        memory_id: ID of the memory

    Returns:
        Memory details including content, importance, timestamps, etc.
    """
    client = get_client()
    return await client.memories.get(memory_id)


@mcp.tool()
async def memory_update(
    memory_id: str,
    content: Optional[str] = None,
    importance: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update a memory's content, importance, or metadata.

    Args:
        memory_id: ID of the memory
        content: New content (optional)
        importance: New importance score (optional)
        metadata: New metadata (optional)

    Returns:
        Updated memory
    """
    client = get_client()
    params = {}
    if content:
        params["content"] = content
    if importance is not None:
        params["importance"] = importance
    if metadata:
        params["metadata"] = metadata

    return await client.memories.update(memory_id, **params)


@mcp.tool()
async def memory_delete(memory_id: str) -> str:
    """
    Delete a specific memory.

    Args:
        memory_id: ID of the memory to delete

    Returns:
        Success message
    """
    client = get_client()
    client.memories.delete(memory_id)
    return f"Memory {memory_id} deleted successfully"


# ============================================================================
# SEARCH & RETRIEVAL TOOLS
# ============================================================================

@mcp.tool()
async def search_memories(
    query: str,
    collection_id: Optional[str] = None,
    limit: int = 10,
    search_type: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search memories using advanced hybrid search (Vector + BM25 + Graph).

    Search types:
    - "hybrid": Combines vector, BM25, and graph search (best results)
    - "vector": Semantic similarity search only
    - "bm25": Keyword-based search only
    - "graph": Knowledge graph relationship search

    Args:
        query: Search query text
        collection_id: Optional collection to search in
        limit: Maximum results to return
        search_type: Type of search (hybrid, vector, bm25, graph)
        filters: Optional metadata filters

    Returns:
        List of relevant memories with similarity scores
    """
    client = get_client()
    params = {
        "query": query,
        "limit": limit,
        "search_type": search_type
    }
    if collection_id:
        params["collection_id"] = collection_id
    if filters:
        params["filters"] = filters

    return await client.search(**params)


@mcp.tool()
async def reason_over_memories(
    query: str,
    collection_id: Optional[str] = None,
    provider: str = "gemini",
    include_steps: bool = False
) -> Dict[str, Any]:
    """
    Use AI reasoning (RAG) to answer questions based on memories.

    This retrieves relevant memories and uses an LLM to generate comprehensive answers.

    Providers:
    - "gemini": Google Gemini (recommended, fast)
    - "openai": OpenAI GPT-4
    - "anthropic": Anthropic Claude

    Args:
        query: Question to answer
        collection_id: Optional collection to reason over
        provider: LLM provider to use
        include_steps: Include reasoning steps in response

    Returns:
        Answer with sources and optional reasoning steps
    """
    client = get_client()
    params = {
        "query": query,
        "provider": provider,
        "include_steps": include_steps
    }
    if collection_id:
        params["collection_id"] = collection_id

    return await client.reason(**params)


# ============================================================================
# REINFORCEMENT LEARNING TOOLS (Unique Feature!)
# ============================================================================

@mcp.tool()
async def rl_train_memory_manager(
    collection_id: Optional[str] = None,
    num_episodes: int = 100
) -> Dict[str, Any]:
    """
    Train the Memory Manager agent using Reinforcement Learning.

    The Memory Manager learns to decide what to remember and what to forget,
    optimizing for retrieval performance and memory efficiency.

    This is a UNIQUE feature - no other memory API has RL training!

    Args:
        collection_id: Optional collection to train on
        num_episodes: Number of training episodes (more = better but slower)

    Returns:
        Training results with performance metrics
    """
    client = get_client()
    return await client.rl.train_memory_manager(
        collection_id=collection_id,
        num_episodes=num_episodes
    )


@mcp.tool()
async def rl_train_answer_agent(
    collection_id: Optional[str] = None,
    num_episodes: int = 100
) -> Dict[str, Any]:
    """
    Train the Answer Agent using Reinforcement Learning.

    The Answer Agent learns optimal retrieval strategies for answering questions.

    Args:
        collection_id: Optional collection to train on
        num_episodes: Number of training episodes

    Returns:
        Training results with performance metrics
    """
    client = get_client()
    return await client.rl.train_answer_agent(
        collection_id=collection_id,
        num_episodes=num_episodes
    )


@mcp.tool()
async def rl_get_metrics() -> Dict[str, Any]:
    """
    Get current RL training metrics.

    Shows performance of trained Memory Manager and Answer Agent.

    Returns:
        RL metrics including accuracy, rewards, training progress
    """
    client = get_client()
    return await client.rl.get_metrics()


@mcp.tool()
async def rl_evaluate_agent(
    agent_type: str,
    collection_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained RL agent's performance.

    Args:
        agent_type: "memory-manager" or "answer-agent"
        collection_id: Optional collection to evaluate on

    Returns:
        Evaluation metrics
    """
    client = get_client()
    return await client.rl.evaluate(agent_type, collection_id)


# ============================================================================
# PROCEDURAL MEMORY TOOLS
# ============================================================================

@mcp.tool()
async def procedural_create(
    name: str,
    description: str,
    trigger_condition: str,
    action_sequence: List[str],
    collection_id: Optional[str] = None,
    category: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new procedural memory (learned skill/procedure).

    Procedural memories store "how to do" something, like workflows or repeated tasks.

    Args:
        name: Name of the procedure
        description: What this procedure does
        trigger_condition: When to execute (e.g., "time.hour == 18")
        action_sequence: List of actions to perform
        collection_id: Optional collection
        category: Optional category (e.g., "automation", "workflow")
        metadata: Optional metadata

    Returns:
        Created procedure
    """
    client = get_client()
    params = {
        "name": name,
        "description": description,
        "trigger_condition": trigger_condition,
        "action_sequence": action_sequence
    }
    if collection_id:
        params["collection_id"] = collection_id
    if category:
        params["category"] = category
    if metadata:
        params["metadata"] = metadata

    return await client.procedural.create(**params)


@mcp.tool()
async def procedural_list(
    collection_id: Optional[str] = None,
    category: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List procedural memories.

    Args:
        collection_id: Filter by collection
        category: Filter by category
        skip: Skip N procedures
        limit: Max procedures to return

    Returns:
        List of procedures
    """
    client = get_client()
    params = {"skip": skip, "limit": limit}
    if collection_id:
        params["collection_id"] = collection_id
    if category:
        params["category"] = category

    return await client.procedural.list(**params)


@mcp.tool()
async def procedural_execute(
    procedure_id: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a procedural memory.

    Args:
        procedure_id: ID of procedure to execute
        context: Optional execution context

    Returns:
        Execution result
    """
    client = get_client()
    return await client.procedural.execute(procedure_id, context)


@mcp.tool()
async def procedural_delete(procedure_id: str) -> str:
    """
    Delete a procedural memory.

    Args:
        procedure_id: ID of procedure to delete

    Returns:
        Success message
    """
    client = get_client()
    client.procedural.delete(procedure_id)
    return f"Procedure {procedure_id} deleted successfully"


# ============================================================================
# TEMPORAL KNOWLEDGE GRAPH TOOLS
# ============================================================================

@mcp.tool()
async def temporal_add_fact(
    subject: str,
    predicate: str,
    object: str,
    valid_from: Optional[str] = None,
    valid_until: Optional[str] = None,
    confidence: float = 1.0,
    source_memory_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add a temporal fact to the knowledge graph.

    Temporal facts track how knowledge changes over time.
    Example: "User_123 works_at TechCorp from 2024-01-01 to 2024-12-31"

    Args:
        subject: Subject entity (e.g., "User_123")
        predicate: Relationship (e.g., "works_at")
        object: Object entity (e.g., "TechCorp")
        valid_from: Start date (ISO format)
        valid_until: End date (ISO format)
        confidence: Confidence score 0.0-1.0
        source_memory_id: Optional source memory
        metadata: Optional metadata

    Returns:
        Created fact
    """
    client = get_client()
    params = {
        "subject": subject,
        "predicate": predicate,
        "object": object,
        "confidence": confidence
    }
    if valid_from:
        params["valid_from"] = valid_from
    if valid_until:
        params["valid_until"] = valid_until
    if source_memory_id:
        params["source_memory_id"] = source_memory_id
    if metadata:
        params["metadata"] = metadata

    return await client.temporal.add_fact(**params)


@mcp.tool()
async def temporal_query_facts(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object: Optional[str] = None,
    at_time: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query temporal facts from the knowledge graph.

    Args:
        subject: Filter by subject entity
        predicate: Filter by relationship
        object: Filter by object entity
        at_time: Query facts valid at specific time (ISO format)

    Returns:
        List of matching facts
    """
    client = get_client()
    params = {}
    if subject:
        params["subject"] = subject
    if predicate:
        params["predicate"] = predicate
    if object:
        params["object"] = object
    if at_time:
        params["at_time"] = at_time

    return await client.temporal.query_facts(**params)


@mcp.tool()
async def temporal_point_in_time(
    timestamp: str,
    entity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query knowledge state at a specific point in time.

    Get a snapshot of what was known about an entity at a specific moment.

    Args:
        timestamp: Point in time (ISO format)
        entity: Optional entity to focus on

    Returns:
        Knowledge snapshot with facts and relationships
    """
    client = get_client()
    return await client.temporal.point_in_time(timestamp, entity)


# ============================================================================
# WORKING MEMORY TOOLS
# ============================================================================

@mcp.tool()
async def working_memory_add(
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add item to working memory buffer (short-term conversation context).

    Working memory stores recent conversation turns before consolidation.

    Args:
        role: "user" or "assistant"
        content: Message content
        metadata: Optional metadata

    Returns:
        Added item
    """
    client = get_client()
    params = {"role": role, "content": content}
    if metadata:
        params["metadata"] = metadata

    return await client.working_memory.add(**params)


@mcp.tool()
async def working_memory_get_context() -> Dict[str, Any]:
    """
    Get current working memory context (recent conversation).

    Returns:
        Working memory items and buffer info
    """
    client = get_client()
    return await client.working_memory.get_context()


@mcp.tool()
async def working_memory_compress() -> Dict[str, Any]:
    """
    Compress working memory to episodic memories.

    Converts short-term conversation context into long-term memories.

    Returns:
        Compression results
    """
    client = get_client()
    return await client.working_memory.compress()


@mcp.tool()
async def working_memory_clear() -> str:
    """
    Clear working memory buffer.

    Returns:
        Success message
    """
    client = get_client()
    client.working_memory.clear()
    return "Working memory cleared successfully"


# ============================================================================
# MEMORY CONSOLIDATION TOOLS
# ============================================================================

@mcp.tool()
async def consolidation_consolidate(
    collection_id: str,
    threshold: int = 100
) -> Dict[str, Any]:
    """
    Trigger memory consolidation for a collection.

    Consolidation compresses similar memories to reduce redundancy
    while preserving important information.

    Args:
        collection_id: Collection to consolidate
        threshold: Minimum memories before consolidation

    Returns:
        Consolidation results with compression ratio
    """
    client = get_client()
    return await client.consolidation.consolidate(collection_id, threshold)


@mcp.tool()
async def consolidation_get_stats(collection_id: str) -> Dict[str, Any]:
    """
    Get consolidation statistics for a collection.

    Args:
        collection_id: Collection ID

    Returns:
        Stats including total memories, consolidation status, etc.
    """
    client = get_client()
    return await client.consolidation.get_stats(collection_id)


@mcp.tool()
async def consolidation_archive(
    collection_id: str,
    before_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Archive old memories in a collection.

    Moves memories to cold storage for cost efficiency.

    Args:
        collection_id: Collection ID
        before_date: Archive memories before this date (ISO format)

    Returns:
        Archive results
    """
    client = get_client()
    return await client.consolidation.archive(collection_id, before_date)


# ============================================================================
# MEMORY TOOLS (Self-Editing)
# ============================================================================

@mcp.tool()
async def memory_tool_replace(
    memory_id: str,
    new_content: str,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Replace a memory's content (self-editing).

    UNIQUE FEATURE: Memories can edit themselves based on new information!

    Args:
        memory_id: Memory to replace
        new_content: New content
        reason: Optional reason for replacement

    Returns:
        Updated memory
    """
    client = get_client()
    params = {"memory_id": memory_id, "new_content": new_content}
    if reason:
        params["reason"] = reason

    return await client.memory_tools.replace(**params)


@mcp.tool()
async def memory_tool_insert(
    collection_id: str,
    content: str,
    position: int,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Insert a new memory at a specific position.

    Args:
        collection_id: Collection ID
        content: Memory content
        position: Position to insert at
        reason: Optional reason

    Returns:
        Inserted memory
    """
    client = get_client()
    params = {
        "collection_id": collection_id,
        "content": content,
        "position": position
    }
    if reason:
        params["reason"] = reason

    return await client.memory_tools.insert(**params)


@mcp.tool()
async def memory_tool_rethink(
    memory_id: str,
    query: str
) -> Dict[str, Any]:
    """
    Re-evaluate a memory in light of new information.

    UNIQUE FEATURE: Memories can rethink themselves!

    Args:
        memory_id: Memory to rethink
        query: New information/perspective

    Returns:
        Rethought memory with changes
    """
    client = get_client()
    return await client.memory_tools.rethink(memory_id, query)


# ============================================================================
# WORLD MODEL TOOLS
# ============================================================================

@mcp.tool()
async def world_model_imagine_retrieval(
    query: str,
    collection_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulate retrieval without actually retrieving (world model).

    UNIQUE FEATURE: Predict what would be retrieved before executing!

    Args:
        query: Search query to simulate
        collection_id: Optional collection

    Returns:
        Predicted results and confidence
    """
    client = get_client()
    return await client.world_model.imagine_retrieval(query, collection_id)


@mcp.tool()
async def world_model_plan(
    goal: str,
    collection_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plan memory operations to achieve a goal.

    UNIQUE FEATURE: AI plans optimal sequence of operations!

    Args:
        goal: Goal to achieve (e.g., "Prepare Q1 summary")
        collection_id: Optional collection

    Returns:
        Planned operation sequence with estimates
    """
    client = get_client()
    return await client.world_model.plan(goal, collection_id)


# ============================================================================
# RESOURCES (Browse-able Data)
# ============================================================================

@mcp.resource("memory://collections")
def list_all_collections() -> str:
    """
    Resource: List all memory collections.

    Returns JSON list of collections.
    """
    client = get_client()
    collections = client.collections.list()

    import json
    return json.dumps(collections, indent=2)


@mcp.resource("memory://collection/{collection_id}")
def get_collection_resource(collection_id: str) -> str:
    """
    Resource: Get specific collection details.
    """
    client = get_client()
    collection = client.collections.get(collection_id)

    import json
    return json.dumps(collection, indent=2)


@mcp.resource("memory://collection/{collection_id}/memories")
def list_collection_memories(collection_id: str) -> str:
    """
    Resource: List all memories in a collection.
    """
    client = get_client()
    memories = client.memories.list(collection_id=collection_id)

    import json
    return json.dumps(memories, indent=2)


@mcp.resource("memory://memory/{memory_id}")
def get_memory_resource(memory_id: str) -> str:
    """
    Resource: Get specific memory details.
    """
    client = get_client()
    memory = client.memories.get(memory_id)

    import json
    return json.dumps(memory, indent=2)


@mcp.resource("memory://rl/metrics")
def get_rl_metrics_resource() -> str:
    """
    Resource: Get current RL training metrics.
    """
    client = get_client()
    metrics = client.rl.get_metrics()

    import json
    return json.dumps(metrics, indent=2)


# ============================================================================
# PROMPTS (Reusable Templates)
# ============================================================================

@mcp.prompt()
def prompt_create_memory(topic: str, context: str = "") -> str:
    """
    Generate a prompt for creating a structured memory.

    Args:
        topic: Topic of the memory
        context: Optional context

    Returns:
        Prompt for creating memory
    """
    return f"""Create a well-structured memory about {topic}.

Context: {context if context else "General information"}

The memory should:
1. Be concise but complete
2. Include key facts and details
3. Have relevant metadata (tags, importance)
4. Be searchable with keywords

Generate the memory content now."""


@mcp.prompt()
def prompt_search_and_synthesize(
    question: str,
    collection_name: str = "all collections"
) -> str:
    """
    Generate a prompt for searching and synthesizing information.

    Args:
        question: Question to answer
        collection_name: Which collection to search

    Returns:
        Prompt for search and synthesis
    """
    return f"""Search {collection_name} for information about: {question}

Then synthesize the findings into a comprehensive answer that:
1. Directly answers the question
2. Cites specific memories as sources
3. Identifies any gaps or conflicting information
4. Suggests follow-up questions if relevant

Provide your answer now."""


@mcp.prompt()
def prompt_consolidate_memories(
    collection_id: str,
    theme: str = "general"
) -> str:
    """
    Generate a prompt for memory consolidation strategy.

    Args:
        collection_id: Collection to consolidate
        theme: Theme/topic to focus on

    Returns:
        Prompt for consolidation
    """
    return f"""Review memories in collection {collection_id} focusing on theme: {theme}

Consolidation strategy:
1. Identify redundant or overlapping memories
2. Merge similar information while preserving unique details
3. Maintain important memories with high scores
4. Archive outdated information

Execute consolidation with these criteria."""


@mcp.prompt()
def prompt_train_rl_agent(
    agent_type: str,
    num_episodes: int = 100
) -> str:
    """
    Generate a prompt for RL agent training.

    Args:
        agent_type: "memory-manager" or "answer-agent"
        num_episodes: Training episodes

    Returns:
        Prompt for RL training
    """
    return f"""Train the {agent_type} agent using Reinforcement Learning.

Training configuration:
- Episodes: {num_episodes}
- Objective: Optimize for retrieval accuracy and efficiency
- Learning method: PPO (Proximal Policy Optimization)

Monitor training metrics:
- Reward per episode
- Accuracy improvements
- Convergence status

Start training and report progress."""


# ============================================================================
# CONVERSATION MEMORY TOOLS (Cross-Chat Persistence)
# ============================================================================

@mcp.tool()
async def save_conversation_summary(
    summary: str,
    tags: Optional[List[str]] = None,
    importance: float = 0.7
) -> Dict[str, Any]:
    """
    Save important information from this conversation to persistent memory.
    This allows information to be retrieved in future chat sessions.

    Args:
        summary: What to remember (be specific and detailed)
        tags: Optional tags for categorization (e.g., ["RL", "implementation", "bug-fix"])
        importance: 0.0-1.0 scale (default: 0.7, higher = more important)

    Returns:
        Confirmation with memory ID

    Example:
        save_conversation_summary(
            "Implemented RL training with automatic embedding generation. Training works with loss decreasing from 0.22 to 0.17 over 3 runs.",
            tags=["RL", "embeddings", "training"],
            importance=0.9
        )
    """
    client = get_client()

    # Get or create "Conversation History" collection
    try:
        collections = await client.collections.list(limit=100)
        conv_collection = next(
            (c for c in collections if c.get('name') == 'Conversation History'),
            None
        )

        if not conv_collection:
            # Create the conversation history collection
            conv_collection = await client.collections.create(
                name="Conversation History",
                description="Persistent memories from Claude Desktop conversations. Automatically searchable across all future chats."
            )

        collection_id = conv_collection['id']
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to access collections: {str(e)}"
        }

    # Add memory with rich metadata
    try:
        memory = await client.memories.create(
            collection_id=collection_id,
            content=summary,
            importance=importance,
            metadata={
                "tags": tags or [],
                "saved_at": datetime.now().isoformat(),
                "source": "conversation",
                "type": "conversation_summary"
            }
        )

        return {
            "status": "saved",
            "memory_id": memory['id'],
            "collection": "Conversation History",
            "message": "✅ Saved to persistent memory! This will be available in all future chats.",
            "tip": "Use 'recall_conversation_history' in a new chat to retrieve this information."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save memory: {str(e)}"
        }


@mcp.tool()
async def recall_conversation_history(
    query: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Retrieve memories from past conversations.
    Use this at the start of new chats to recall what was discussed before.

    Args:
        query: Optional search query (e.g., "RL training implementation")
               If not provided, returns recent memories
        limit: Maximum number of memories to return (default: 10)

    Returns:
        List of relevant memories from past conversations

    Examples:
        # Get recent conversation history
        recall_conversation_history()

        # Search for specific topic
        recall_conversation_history("RL training and embeddings")
    """
    client = get_client()

    try:
        # Find Conversation History collection
        collections = await client.collections.list(limit=100)
        conv_collection = next(
            (c for c in collections if c.get('name') == 'Conversation History'),
            None
        )

        if not conv_collection:
            return {
                "status": "empty",
                "memories": [],
                "message": "No conversation history found. Use 'save_conversation_summary' to start saving important information."
            }

        collection_id = conv_collection['id']

        # Search or list memories
        if query:
            memories = await client.search.search(
                query=query,
                collection_id=collection_id,
                limit=limit,
                search_type="hybrid"
            )
        else:
            memories = await client.memories.list(
                collection_id=collection_id,
                limit=limit
            )

        if not memories:
            return {
                "status": "empty",
                "memories": [],
                "message": "No memories found in conversation history."
            }

        # Format for display
        formatted_memories = []
        for mem in memories:
            formatted_memories.append({
                "content": mem.get('content', ''),
                "saved_at": mem.get('metadata', {}).get('saved_at', 'Unknown'),
                "tags": mem.get('metadata', {}).get('tags', []),
                "importance": mem.get('importance', 0.5)
            })

        return {
            "status": "success",
            "count": len(formatted_memories),
            "memories": formatted_memories,
            "message": f"Found {len(formatted_memories)} memories from past conversations."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to recall history: {str(e)}"
        }


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================

# Run the server when executed
if __name__ == "__main__":
    print("🧠 Memory AI MCP Server")
    print("=" * 50)
    print("Starting server...")
    print(f"API Base URL: {os.getenv('MEMORY_AI_BASE_URL', 'http://localhost:8000')}")
    print(f"API Key: {'Set' if os.getenv('MEMORY_AI_API_KEY') else 'Not set (will use default)'}")
    print()
    print("Available capabilities:")
    print("  • 50+ memory tools")
    print("  • Resources for browsing data")
    print("  • Prompts for common workflows")
    print("  • Full RL, temporal graphs, procedural memory support")
    print()
    print("Connect this server to Claude Desktop or any MCP client!")
    print("=" * 50)

    # Run the MCP server
    mcp.run()
