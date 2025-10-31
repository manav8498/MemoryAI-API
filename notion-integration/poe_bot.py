#!/usr/bin/env python3.11
"""
Poe Server Bot for Memory AI
Allows Poe users to search your Notion memories
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv
import json
from typing import AsyncIterable
import google.generativeai as genai

# Load env
load_dotenv("/Users/manavpatel/Documents/API Memory/.env")

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

MEMORY_AI_API_KEY = os.getenv("MEMORY_AI_API_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiMWJjNzVlOS0yMzdkLTRiNWMtYWZmOC03ZGFhMWUwN2M1YTYiLCJleHAiOjE3OTMzOTgzNDMsImlhdCI6MTc2MTg2MjM0MywidHlwZSI6ImFjY2VzcyJ9.qOHp8720i8scA0IpEAFMZtQETcFmqmv7mk-pGukeL7A"
MEMORY_AI_BASE_URL = "http://localhost:8000"
COLLECTION_ID = "32ab4192-241a-48fc-9051-6829246b0ca7"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_response(text: str) -> AsyncIterable[str]:
    """Stream response in Poe Bot API format"""
    # Send meta event
    yield f"event: meta\ndata: {json.dumps({'content_type': 'text/markdown', 'linkify': True, 'suggested_replies': False})}\n\n"

    # Send text event
    yield f"event: text\ndata: {json.dumps({'text': text})}\n\n"

    # Send done event
    yield f"event: done\ndata: {{}}\n\n"

@app.post("/")
async def poe_bot_endpoint(request: Request):
    """Main Poe bot endpoint"""
    try:
        data = await request.json()

        # LOG: See what Poe is sending
        print(f"\n📥 Received request type: {data.get('type')}")

        # Handle Poe settings request (for verification)
        if data.get("type") == "settings":
            return JSONResponse({
                "server_bot_dependencies": {},
                "allow_attachments": False,
                "introduction_message": "Hi! I'm your AI-powered Notion assistant. I can search your workspace and answer questions about your notes, job applications, and memories in a natural, conversational way. What would you like to know?",
                "enforce_author_role_alternation": False,
                "enable_image_comprehension": False,
                "expand_text_attachments": False,
                "enable_multi_bot_chat_prompting": False
            })

        # Handle error reports from Poe
        if data.get("type") == "report_error":
            print(f"⚠️  Poe reported error: {data.get('message')}")
            return JSONResponse({"success": True})

        # Extract user query from Poe format
        query_list = data.get("query", [])
        if not query_list:
            return StreamingResponse(
                stream_response("No query received"),
                media_type="text/event-stream"
            )

        user_query = query_list[-1].get("content", "")

        if not user_query:
            return StreamingResponse(
                stream_response("Please ask me something about your Notion data!"),
                media_type="text/event-stream"
            )

        # Detect if user is asking for a list/overview of all pages
        listing_keywords = ["all pages", "overview", "how many", "list all", "what pages", "workspace pages", "total pages"]
        is_listing_query = any(keyword in user_query.lower() for keyword in listing_keywords)

        # Use higher top_k for listing queries to get all pages
        top_k = 100 if is_listing_query else 10

        # Search Memory AI
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEMORY_AI_BASE_URL}/v1/search",
                json={
                    "query": user_query if not is_listing_query else "application",  # Generic query for listing
                    "collection_id": COLLECTION_ID,
                    "limit": top_k  # API uses "limit" not "top_k"
                },
                headers={
                    "Authorization": f"Bearer {MEMORY_AI_API_KEY}",
                    "ngrok-skip-browser-warning": "true"
                },
                timeout=30.0
            )

        if response.status_code != 200:
            return StreamingResponse(
                stream_response(f"❌ Memory AI API error: {response.status_code}"),
                media_type="text/event-stream"
            )

        results = response.json().get("results", [])

        if not results:
            return StreamingResponse(
                stream_response("I couldn't find any relevant information in your Notion pages about that."),
                media_type="text/event-stream"
            )

        # Compile context from search results
        # For listing queries, use ALL results; for specific queries, use top 5
        max_results = len(results) if is_listing_query else 5
        context_parts = []

        if is_listing_query:
            # For listing queries, deduplicate by title and just include unique page titles
            seen_titles = {}
            for result in results:
                metadata = result.get("metadata", {})
                title = metadata.get("title", "Untitled")
                if title not in seen_titles:
                    seen_titles[title] = result

            # Now create context from unique titles
            for i, (title, result) in enumerate(seen_titles.items(), 1):
                content_preview = result.get("content", "")[:100]
                context_parts.append(f"{i}. {title}\n   Preview: {content_preview}...")
        else:
            # For specific queries, include full content
            for i, result in enumerate(results[:max_results], 1):
                content = result.get("content", "")
                context_parts.append(f"--- Page {i} ---\n{content}\n")

        context = "\n".join(context_parts)

        # Generate natural response using Gemini AI
        if is_listing_query:
            unique_page_count = len(seen_titles)
            prompt = f"""You are a helpful assistant with access to the user's Notion workspace data.

The user asked: "{user_query}"

Here is a complete list of ALL unique pages in their Notion workspace ({unique_page_count} pages total):

{context}

Instructions:
- List ALL {unique_page_count} unique pages from the workspace
- Organize them in a clear, numbered list
- Include the page title for each entry
- Group by category if possible (e.g., job applications, meeting notes, etc.)
- Start with the total count: "You have {unique_page_count} pages in your Notion workspace:"
- Use markdown formatting for better readability"""
        else:
            prompt = f"""You are a helpful assistant with access to the user's Notion workspace data.

The user asked: "{user_query}"

Here is the relevant information from their Notion pages:

{context}

Instructions:
- Answer the question naturally and conversationally like ChatGPT
- Focus on directly answering what they asked
- Organize information clearly with bullet points or numbering when appropriate
- Don't mention "pages" or "search results" - just provide the information
- If there are dates, format them nicely
- Keep response concise but complete
- Use markdown formatting for better readability"""

        try:
            gemini_response = gemini_model.generate_content(prompt)
            answer = gemini_response.text
        except Exception as gemini_error:
            print(f"⚠️  Gemini error: {gemini_error}")
            # Fallback to basic formatting if Gemini fails
            answer = "Based on your Notion data:\n\n"
            for i, result in enumerate(results[:3], 1):
                content = result.get("content", "")
                lines = content.split("\n")
                title = lines[0].strip("# ") if lines else "Untitled"
                answer += f"**{i}. {title}**\n"
                for line in lines[1:5]:
                    if line.strip():
                        answer += f"{line}\n"
                answer += "\n"

        query_type = "LISTING ALL PAGES" if is_listing_query else "SEARCH"
        result_count = len(seen_titles) if is_listing_query else len(results)
        print(f"📤 [{query_type}] Sending AI-generated response with {result_count} unique pages\n")
        return StreamingResponse(
            stream_response(answer),
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
    return {"status": "ok", "bot": "Memory AI Poe Bot"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Poe Bot Server on port 8080...")
    print("Your bot will be available at: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
