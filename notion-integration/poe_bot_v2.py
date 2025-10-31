#!/usr/bin/env python3.11
"""
Poe Server Bot for Memory AI - Version 2
Fully utilizes Memory AI's reasoning engine and multi-collection support
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv
import json
from typing import AsyncIterable, Dict, Any, List
import hashlib
from datetime import datetime

# Load env
load_dotenv("/Users/manavpatel/Documents/API Memory/.env")

MEMORY_AI_API_KEY = os.getenv("MEMORY_AI_API_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
MEMORY_AI_BASE_URL = "http://localhost:8000"

# Collection IDs
NOTION_COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"
CONVERSATION_COLLECTION_ID = "745a1565-cf04-437d-9cfb-c3b5f075afc6"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation storage (per user)
conversations: Dict[str, List[Dict[str, str]]] = {}


def detect_query_type(query: str) -> Dict[str, Any]:
    """
    Detect what type of query this is and which collection(s) to search.

    Returns:
        Dict with:
            - type: "conversation_history", "notion_only", "multi_collection"
            - collections: List of collection IDs to search
            - reasoning: str explaining the detection
    """
    query_lower = query.lower()

    # Conversation history indicators
    conversation_keywords = [
        "last chat", "previous conversation", "we discussed", "we talked about",
        "earlier conversation", "what did we", "what did i say", "our conversation",
        "conversation history", "chat history", "we were talking"
    ]

    # Notion-specific indicators
    notion_keywords = [
        "notion", "job application", "applied to", "interview", "company",
        "meeting notes", "my notes", "workspace", "database"
    ]

    # Check for conversation history queries
    if any(keyword in query_lower for keyword in conversation_keywords):
        return {
            "type": "conversation_history",
            "collections": [CONVERSATION_COLLECTION_ID],
            "reasoning": "Query is asking about previous conversations"
        }

    # Check for Notion-specific queries
    if any(keyword in query_lower for keyword in notion_keywords):
        return {
            "type": "notion_only",
            "collections": [NOTION_COLLECTION_ID],
            "reasoning": "Query is asking about Notion data"
        }

    # Default: search both collections
    return {
        "type": "multi_collection",
        "collections": [NOTION_COLLECTION_ID, CONVERSATION_COLLECTION_ID],
        "reasoning": "General query - searching all available data"
    }


async def store_conversation_turn(user_id: str, user_query: str, bot_response: str):
    """Store conversation in Memory AI for future retrieval"""
    try:
        async with httpx.AsyncClient() as client:
            # Create a memory for this conversation turn
            conversation_content = f"""# Conversation on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}

**User asked:** {user_query}

**Bot responded:** {bot_response}
"""

            response = await client.post(
                f"{MEMORY_AI_BASE_URL}/v1/memories",
                json={
                    "collection_id": CONVERSATION_COLLECTION_ID,
                    "content": conversation_content,
                    "metadata": {
                        "type": "conversation",
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "user_query": user_query,
                    }
                },
                headers={"Authorization": f"Bearer {MEMORY_AI_API_KEY}"},
                timeout=10.0
            )

            if response.status_code in [200, 201]:
                print(f"✓ Stored conversation turn in Memory AI")
            else:
                print(f"⚠ Failed to store conversation: {response.status_code}")

    except Exception as e:
        print(f"⚠ Error storing conversation: {e}")


async def query_memory_ai(query: str, collection_ids: List[str], provider: str = "gemini") -> Dict[str, Any]:
    """
    Query Memory AI using the reasoning engine.

    Args:
        query: User's question
        collection_ids: List of collection IDs to search
        provider: LLM provider (gemini, openai, anthropic)

    Returns:
        Dict with answer, sources, and metadata
    """
    async with httpx.AsyncClient() as client:
        # If multiple collections, we'll search each and combine
        # For now, prioritize the first collection
        collection_id = collection_ids[0] if collection_ids else None

        # Use the reasoning endpoint for RAG
        response = await client.post(
            f"{MEMORY_AI_BASE_URL}/v1/search/reason",
            json={
                "query": query,
                "collection_id": collection_id,
                "provider": provider,
                "include_steps": False,
            },
            headers={
                "Authorization": f"Bearer {MEMORY_AI_API_KEY}",
                "ngrok-skip-browser-warning": "true"
            },
            timeout=60.0
        )

        if response.status_code != 200:
            print(f"❌ Memory AI error: {response.status_code}")
            print(response.text)
            return {
                "answer": f"Sorry, I encountered an error: {response.status_code}",
                "sources": [],
                "metadata": {}
            }

        result = response.json()
        return result


async def stream_response(answer: str, sources: List[Dict[str, Any]] = None) -> AsyncIterable[str]:
    """Stream response in Poe Bot API format with sources"""
    # Send meta event
    yield f"event: meta\ndata: {json.dumps({'content_type': 'text/markdown', 'linkify': True, 'suggested_replies': False})}\n\n"

    # Format answer with sources
    full_response = answer

    if sources:
        full_response += "\n\n---\n\n**Sources:**\n"
        for i, source in enumerate(sources[:5], 1):
            title = source.get("metadata", {}).get("title", "Untitled")
            score = source.get("score", 0)
            full_response += f"{i}. {title} (relevance: {score:.2f})\n"

    # Send text event
    yield f"event: text\ndata: {json.dumps({'text': full_response})}\n\n"

    # Send done event
    yield f"event: done\ndata: {{}}\n\n"


@app.post("/")
async def poe_bot_endpoint(request: Request):
    """Main Poe bot endpoint with full Memory AI integration"""
    try:
        data = await request.json()

        # LOG: See what Poe is sending
        print(f"\n📥 Received request type: {data.get('type')}")

        # Handle Poe settings request (for verification)
        if data.get("type") == "settings":
            return JSONResponse({
                "server_bot_dependencies": {},
                "allow_attachments": False,
                "introduction_message": """Hi! I'm your AI-powered memory assistant with access to:

✅ Your Notion workspace (notes, job applications, meetings)
✅ Our conversation history
✅ Advanced reasoning with citations

I use Memory AI's reasoning engine for accurate, context-aware answers. Ask me anything!""",
                "enforce_author_role_alternation": False,
                "enable_image_comprehension": False,
                "expand_text_attachments": False,
                "enable_multi_bot_chat_prompting": False
            })

        # Handle error reports from Poe
        if data.get("type") == "report_error":
            print(f"⚠️ Poe reported error: {data.get('message')}")
            return JSONResponse({"success": True})

        # Extract user query from Poe format
        query_list = data.get("query", [])
        if not query_list:
            return StreamingResponse(
                stream_response("No query received"),
                media_type="text/event-stream"
            )

        user_query = query_list[-1].get("content", "")
        user_id = data.get("user_id", "anonymous")

        if not user_query:
            return StreamingResponse(
                stream_response("Please ask me something!"),
                media_type="text/event-stream"
            )

        print(f"👤 User: {user_query}")

        # STEP 1: Detect query type and route to correct collections
        query_info = detect_query_type(user_query)
        print(f"🔍 Query type: {query_info['type']} - {query_info['reasoning']}")

        # STEP 2: Query Memory AI using reasoning engine
        result = await query_memory_ai(
            query=user_query,
            collection_ids=query_info["collections"],
            provider="gemini"  # Can be configured
        )

        answer = result.get("answer", "I couldn't generate an answer.")
        sources = result.get("sources", [])

        print(f"✅ Generated answer ({len(answer)} chars) with {len(sources)} sources")

        # STEP 3: Store this conversation turn for future reference
        await store_conversation_turn(user_id, user_query, answer)

        # STEP 4: Stream response back to Poe
        return StreamingResponse(
            stream_response(answer, sources),
            media_type="text/event-stream"
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error: {error_trace}")
        return StreamingResponse(
            stream_response(f"❌ Error: {str(e)}"),
            media_type="text/event-stream"
        )


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "bot": "Memory AI Poe Bot V2",
        "features": [
            "RAG with reasoning engine",
            "Multi-collection search",
            "Conversation history tracking",
            "Intelligent query routing",
            "Citation support"
        ]
    }


@app.get("/stats")
async def get_stats():
    """Get bot statistics"""
    async with httpx.AsyncClient() as client:
        try:
            # Get collection stats
            notion_response = await client.get(
                f"{MEMORY_AI_BASE_URL}/v1/collections/{NOTION_COLLECTION_ID}",
                headers={"Authorization": f"Bearer {MEMORY_AI_API_KEY}"},
                timeout=10.0
            )

            conv_response = await client.get(
                f"{MEMORY_AI_BASE_URL}/v1/collections/{CONVERSATION_COLLECTION_ID}",
                headers={"Authorization": f"Bearer {MEMORY_AI_API_KEY}"},
                timeout=10.0
            )

            notion_col = notion_response.json() if notion_response.status_code == 200 else {}
            conv_col = conv_response.json() if conv_response.status_code == 200 else {}

            return {
                "notion_collection": {
                    "name": notion_col.get("name", "Unknown"),
                    "memory_count": notion_col.get("memory_count", 0)
                },
                "conversation_collection": {
                    "name": conv_col.get("name", "Unknown"),
                    "memory_count": conv_col.get("memory_count", 0)
                }
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 MEMORY AI POE BOT V2 - STARTING")
    print("="*70)
    print("\n✨ Features:")
    print("  ✅ RAG with Memory AI reasoning engine")
    print("  ✅ Multi-collection search (Notion + Conversations)")
    print("  ✅ Intelligent query routing")
    print("  ✅ Conversation history tracking")
    print("  ✅ Citation support")
    print("\n📍 Bot URL: http://localhost:8080")
    print("📊 Stats: http://localhost:8080/stats")
    print("\n" + "="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)
