#!/usr/bin/env python3.11
"""
Smart Chat with Your Notion Data
Answers questions naturally using AI (like ChatGPT)
"""
import asyncio
import os
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot
import google.generativeai as genai

# Load both env files
load_dotenv()
load_dotenv("/Users/manavpatel/Documents/API Memory/.env")

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

async def smart_chat():
    """Chat that answers naturally using AI"""

    print("\n" + "="*60)
    print("💬 SMART CHAT - Talk to Your Notion Data")
    print("="*60 + "\n")

    bot = NotionMemoryBot()
    await bot.initialize()

    # Check how many memories we have
    try:
        response = await bot.memory_client.get(
            f"/v1/collections/{bot.collection_id}"
        )
        memory_count = response.json().get("memory_count", 0)
        print(f"📊 Connected to {memory_count} Notion pages\n")

        if memory_count == 0:
            print("⚠️  No Notion data synced yet!")
            print("   Run: python3.11 auto_sync_all.py first\n")
            await bot.close()
            return

    except Exception as e:
        print(f"⚠️  Could not check memories: {e}\n")

    print("Ask me anything about your Notion data!")
    print("I'll answer naturally like ChatGPT.\n")
    print("Examples:")
    print('  - "When did I apply to CVS Health?"')
    print('  - "What interviews do I have coming up?"')
    print('  - "Which companies haven\'t responded yet?"')
    print('  - "Summarize my job search status"')
    print("")
    print("Type 'quit' or 'exit' to stop\n")
    print("="*60 + "\n")

    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    while True:
        try:
            # Get user question
            question = input("\n🧑 You: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            # Search for relevant information
            print("\n🤔 Thinking...", end='', flush=True)

            results = await bot.search_memories(question, top_k=5)

            if not results:
                print("\n\n🤖 AI: I couldn't find any relevant information in your Notion pages about that.\n")
                continue

            # Compile context from search results
            context_parts = []
            for i, result in enumerate(results, 1):
                content = result.get("content", "")
                score = result.get("score", 0)

                # Only include highly relevant results
                if score > 0.005:
                    context_parts.append(f"Document {i}:\n{content}\n")

            context = "\n".join(context_parts)

            # Create prompt for Gemini
            prompt = f"""You are a helpful assistant with access to the user's Notion workspace data.

The user asked: "{question}"

Here is the relevant information from their Notion pages:

{context}

Based ONLY on the information above, answer the user's question naturally and conversationally.

Important:
- Be direct and concise
- Use the actual data from the documents (dates, company names, etc.)
- If asked "when", provide the specific date
- If asked about status, tell them the current status
- Format your answer naturally like ChatGPT would
- Don't mention "documents" or "search results" - just answer as if you know this information
- If the information isn't in the documents, say so

Answer:"""

            # Get AI response
            response = model.generate_content(prompt)
            answer = response.text

            print(f"\r🤖 AI: {answer}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()

    await bot.close()

if __name__ == "__main__":
    asyncio.run(smart_chat())
