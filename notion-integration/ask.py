#!/usr/bin/env python3.11
"""
Simple script to ask questions about your Notion notes
"""
import asyncio
import sys
from dotenv import load_dotenv
from notion_memory_bot import NotionMemoryBot

load_dotenv()

async def ask_question(question: str):
    """Ask a question about your Notion notes"""

    print("\n" + "="*60)
    print("ASKING MEMORY AI ABOUT YOUR NOTION NOTES")
    print("="*60 + "\n")

    # Initialize bot
    bot = NotionMemoryBot()
    await bot.initialize()

    # Ask question
    print(f"❓ Question: {question}\n")
    print("🤔 Thinking...\n")

    answer = await bot.ask_question(question)

    print("="*60)
    print("💡 ANSWER:")
    print("="*60)
    print(f"\n{answer}\n")

    await bot.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Error: Question required")
        print("\nUsage:")
        print('  python3.11 ask.py "Your question here"')
        print("\nExamples:")
        print('  python3.11 ask.py "What are my top priorities?"')
        print('  python3.11 ask.py "What did I learn about Python?"')
        print('  python3.11 ask.py "Show me all meeting notes"')
        print("")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    asyncio.run(ask_question(question))
