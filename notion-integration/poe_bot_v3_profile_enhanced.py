#!/usr/bin/env python3.11
"""
Poe Server Bot for Memory AI - Version 3 (Profile-Enhanced)

Features:
✨ Full user profile integration for personalized responses
✨ Profile-based reranking of search results
✨ Personalized conversation starters
✨ Context-aware responses based on expertise and interests
✨ Performance optimized with caching
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv
import json
from typing import AsyncIterable, Dict, Any, List, Optional
from datetime import datetime
import asyncio

# Load env
load_dotenv("/Users/manavpatel/Documents/API Memory/.env")

MEMORY_AI_API_KEY = os.getenv("MEMORY_AI_API_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
MEMORY_AI_BASE_URL = "http://localhost:8000"

# Collection IDs
NOTION_COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"
CONVERSATION_COLLECTION_ID = "745a1565-cf04-437d-9cfb-c3b5f075afc6"

# Profile cache (TTL: 5 minutes)
profile_cache: Dict[str, Dict[str, Any]] = {}
PROFILE_CACHE_TTL = 300

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user profile with caching.

    Args:
        user_id: User ID

    Returns:
        User profile or None
    """
    # Check cache
    if user_id in profile_cache:
        cached = profile_cache[user_id]
        age = datetime.utcnow().timestamp() - cached["cached_at"]
        if age < PROFILE_CACHE_TTL:
            print(f"📦 Using cached profile for {user_id}")
            return cached["profile"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEMORY_AI_BASE_URL}/v1/profile",
                json={
                    "include_static": True,
                    "include_dynamic": True,
                    "min_confidence": 0.5
                },
                headers={"Authorization": f"Bearer {MEMORY_AI_API_KEY}"},
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                profile = data.get("profile", {})

                # Cache it
                profile_cache[user_id] = {
                    "profile": profile,
                    "cached_at": datetime.utcnow().timestamp()
                }

                print(f"✓ Retrieved profile: {profile['metadata']['total_facts']} facts")
                return profile

            print(f"⚠ Profile retrieval failed: {response.status_code}")
            return None

    except Exception as e:
        print(f"⚠ Error getting profile: {e}")
        return None


async def get_conversation_starters(user_id: str, count: int = 3) -> List[str]:
    """
    Get personalized conversation starters.

    Args:
        user_id: User ID
        count: Number of starters

    Returns:
        List of conversation starter questions
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEMORY_AI_BASE_URL}/v1/profile/starters",
                params={"count": count},
                headers={"Authorization": f"Bearer {MEMORY_AI_API_KEY}"},
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                return [s["question"] for s in data.get("starters", [])]

            return []

    except Exception as e:
        print(f"⚠ Error getting starters: {e}")
        return []


def build_profile_context(profile: Dict[str, Any]) -> str:
    """
    Build a concise profile context string for LLM.

    Args:
        profile: User profile

    Returns:
        Formatted context string
    """
    if not profile:
        return ""

    context_parts = []

    # Add static facts (expertise, role, etc.)
    static_facts = profile.get("static", [])[:5]  # Limit to top 5
    if static_facts:
        context_parts.append("**User Profile:**")
        for fact in static_facts:
            context_parts.append(f"- {fact['key']}: {fact['value']}")

    # Add dynamic facts (current projects, recent skills)
    dynamic_facts = profile.get("dynamic", [])[:3]  # Limit to top 3
    if dynamic_facts:
        context_parts.append("\n**Current Context:**")
        for fact in dynamic_facts:
            context_parts.append(f"- {fact['key']}: {fact['value']}")

    if context_parts:
        return "\n".join(context_parts) + "\n"

    return ""


async def rerank_with_profile(
    results: List[Dict[str, Any]],
    profile: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Rerank search results based on user profile.

    Boosts results that match user's:
    - Expertise
    - Current projects
    - Interests
    - Goals

    Args:
        results: Search results
        profile: User profile

    Returns:
        Reranked results
    """
    if not profile or not results:
        return results

    # Extract profile keywords
    keywords = []
    for fact in profile.get("static", []) + profile.get("dynamic", []):
        value = fact.get("value", "").lower()
        # Simple tokenization
        keywords.extend(value.split())

    # Remove duplicates and filter short words
    keywords = list(set([k for k in keywords if len(k) > 3]))

    # Score and rerank
    for result in results:
        content = result.get("content", "").lower()
        original_score = result.get("score", 0.0)

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in content)

        # Apply boost based on matches
        if matches > 0:
            boost = 1.0 + (0.1 * matches)  # 10% boost per match
            result["original_score"] = original_score
            result["profile_boost"] = boost
            result["score"] = original_score * boost

    # Sort by boosted score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


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


async def query_memory_ai_with_profile(
    query: str,
    collection_ids: List[str],
    profile: Optional[Dict[str, Any]] = None,
    provider: str = "gemini"
) -> Dict[str, Any]:
    """
    Query Memory AI with profile context.

    Args:
        query: User's question
        collection_ids: List of collection IDs to search
        profile: User profile (optional)
        provider: LLM provider (gemini, openai, anthropic)

    Returns:
        Dict with answer, sources, and metadata
    """
    async with httpx.AsyncClient() as client:
        collection_id = collection_ids[0] if collection_ids else None

        # Build profile-enhanced query
        enhanced_query = query
        if profile:
            profile_context = build_profile_context(profile)
            if profile_context:
                enhanced_query = f"{profile_context}\n\n**User Question:** {query}"

        print(f"🔍 Enhanced query with profile context: {len(enhanced_query)} chars")

        # Use the reasoning endpoint for RAG
        response = await client.post(
            f"{MEMORY_AI_BASE_URL}/v1/search/reason",
            json={
                "query": enhanced_query,
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

        # Apply profile-based reranking to sources
        if profile and result.get("sources"):
            print(f"🎯 Applying profile-based reranking to {len(result['sources'])} sources")
            result["sources"] = await rerank_with_profile(result["sources"], profile)

        return result


async def store_conversation_turn(user_id: str, user_query: str, bot_response: str):
    """Store conversation in Memory AI for future retrieval"""
    try:
        async with httpx.AsyncClient() as client:
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


async def stream_response(
    answer: str,
    sources: List[Dict[str, Any]] = None,
    profile_enhanced: bool = False
) -> AsyncIterable[str]:
    """Stream response in Poe Bot API format with sources"""
    # Send meta event
    yield f"event: meta\ndata: {json.dumps({'content_type': 'text/markdown', 'linkify': True, 'suggested_replies': False})}\n\n"

    # Format answer with sources
    full_response = answer

    if profile_enhanced:
        full_response = "✨ *Personalized response based on your profile*\n\n" + full_response

    if sources:
        full_response += "\n\n---\n\n**Sources:**\n"
        for i, source in enumerate(sources[:5], 1):
            title = source.get("metadata", {}).get("title", "Untitled")
            score = source.get("score", 0)
            boost = source.get("profile_boost")

            source_line = f"{i}. {title} (relevance: {score:.2f}"
            if boost and boost > 1.0:
                source_line += f", profile-boosted: {boost:.2f}x"
            source_line += ")\n"

            full_response += source_line

    # Send text event
    yield f"event: text\ndata: {json.dumps({'text': full_response})}\n\n"

    # Send done event
    yield f"event: done\ndata: {{}}\n\n"


@app.post("/")
async def poe_bot_endpoint(request: Request):
    """Main Poe bot endpoint with full profile integration"""
    try:
        data = await request.json()

        print(f"\n📥 Received request type: {data.get('type')}")

        # Handle Poe settings request
        if data.get("type") == "settings":
            # Get sample starters for introduction
            starters = await get_conversation_starters("b1bc75e9-237d-4b5c-aff8-7daa1e07c5a6", count=3)
            intro_starters = "\n".join(f"• {s}" for s in starters[:3]) if starters else ""

            intro_message = f"""Hi! I'm your AI-powered memory assistant with **profile-aware personalization**! 🎯

✅ Your Notion workspace (notes, job applications, meetings)
✅ Our conversation history
✅ Advanced reasoning with citations
✨ **Personalized responses based on your expertise and interests**
✨ **Smart result ranking based on what matters to you**

{f"**Suggested topics based on your profile:**\n{intro_starters}" if intro_starters else ""}

Ask me anything!"""

            return JSONResponse({
                "server_bot_dependencies": {},
                "allow_attachments": False,
                "introduction_message": intro_message,
                "enforce_author_role_alternation": False,
                "enable_image_comprehension": False,
                "expand_text_attachments": False,
                "enable_multi_bot_chat_prompting": False
            })

        # Handle error reports
        if data.get("type") == "report_error":
            print(f"⚠️ Poe reported error: {data.get('message')}")
            return JSONResponse({"success": True})

        # Extract user query
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

        # STEP 1: Get user profile (with caching)
        print(f"📋 Retrieving user profile...")
        profile = await get_user_profile(user_id)

        # STEP 2: Detect query type and route
        query_info = detect_query_type(user_query)
        print(f"🔍 Query type: {query_info['type']} - {query_info['reasoning']}")

        # STEP 3: Query Memory AI with profile context
        result = await query_memory_ai_with_profile(
            query=user_query,
            collection_ids=query_info["collections"],
            profile=profile,
            provider="gemini"
        )

        answer = result.get("answer", "I couldn't generate an answer.")
        sources = result.get("sources", [])

        print(f"✅ Generated answer ({len(answer)} chars) with {len(sources)} sources")
        if profile:
            print(f"✨ Profile-enhanced response")

        # STEP 4: Store conversation
        await store_conversation_turn(user_id, user_query, answer)

        # STEP 5: Stream response
        return StreamingResponse(
            stream_response(answer, sources, profile_enhanced=bool(profile)),
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
        "bot": "Memory AI Poe Bot V3 (Profile-Enhanced)",
        "features": [
            "RAG with reasoning engine",
            "Multi-collection search",
            "Conversation history tracking",
            "Intelligent query routing",
            "Citation support",
            "✨ Profile-based personalization",
            "✨ Profile-aware reranking",
            "✨ Personalized conversation starters",
            "✨ Context-aware responses"
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
                },
                "profile_cache_size": len(profile_cache)
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 MEMORY AI POE BOT V3 - PROFILE-ENHANCED")
    print("="*70)
    print("\n✨ New Features:")
    print("  🎯 Profile-based personalization")
    print("  📊 Profile-aware result reranking")
    print("  💬 Personalized conversation starters")
    print("  ⚡ Performance optimized with caching")
    print("\n✅ Core Features:")
    print("  ✅ RAG with Memory AI reasoning engine")
    print("  ✅ Multi-collection search (Notion + Conversations)")
    print("  ✅ Intelligent query routing")
    print("  ✅ Conversation history tracking")
    print("  ✅ Citation support")
    print("\n📍 Bot URL: http://localhost:8080")
    print("📊 Stats: http://localhost:8080/stats")
    print("\n" + "="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)
