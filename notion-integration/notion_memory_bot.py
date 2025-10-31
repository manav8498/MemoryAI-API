#!/usr/bin/env python3.11
"""
Notion Memory AI Integration Bot
Syncs Notion pages with Memory AI for intelligent memory management
"""
import os
import asyncio
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx
from notion_client import AsyncClient as NotionClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
MEMORY_AI_API_KEY = os.getenv("MEMORY_AI_API_KEY")
MEMORY_AI_BASE_URL = os.getenv("MEMORY_AI_BASE_URL", "http://localhost:8000")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")  # Optional: specific database to sync

# Initialize clients
notion = NotionClient(auth=NOTION_TOKEN)


class NotionMemoryBot:
    """Bot that syncs Notion pages with Memory AI"""

    def __init__(self):
        self.notion = NotionClient(auth=NOTION_TOKEN)
        self.memory_client = httpx.AsyncClient(
            base_url=MEMORY_AI_BASE_URL,
            headers={"Authorization": f"Bearer {MEMORY_AI_API_KEY}"},
            timeout=30.0
        )
        self.collection_id = None

    async def initialize(self):
        """Initialize Memory AI collection for Notion"""
        print("🚀 Initializing Notion Memory Bot...")

        # Get or create "Notion Notes" collection
        try:
            response = await self.memory_client.get("/v1/collections")
            collections = response.json()

            # Find existing collection
            for col in collections:
                if col["name"] == "Notion Notes":
                    self.collection_id = col["id"]
                    print(f"✓ Using existing collection: {self.collection_id}")
                    break

            # Create if doesn't exist
            if not self.collection_id:
                response = await self.memory_client.post(
                    "/v1/collections",
                    json={
                        "name": "Notion Notes",
                        "description": "Memories synced from Notion pages",
                        "metadata": {"source": "notion", "auto_sync": True}
                    }
                )
                self.collection_id = response.json()["id"]
                print(f"✓ Created new collection: {self.collection_id}")

        except Exception as e:
            print(f"✗ Failed to initialize Memory AI: {e}")
            raise

        print("✓ Bot initialized successfully!\n")

    async def extract_block_text(self, block: Dict[str, Any], indent_level: int = 0) -> str:
        """Extract text from a single block, including nested children"""
        block_type = block.get("type", "unsupported")
        text_parts = []
        indent = "  " * indent_level  # Indentation for nested blocks

        # Handle all block types
        try:
            if block_type == "paragraph":
                rich_text = block["paragraph"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}{text}")

            elif block_type == "heading_1":
                rich_text = block["heading_1"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}# {text}")

            elif block_type == "heading_2":
                rich_text = block["heading_2"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}## {text}")

            elif block_type == "heading_3":
                rich_text = block["heading_3"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}### {text}")

            elif block_type == "bulleted_list_item":
                rich_text = block["bulleted_list_item"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}• {text}")

            elif block_type == "numbered_list_item":
                rich_text = block["numbered_list_item"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}- {text}")

            elif block_type == "to_do":
                rich_text = block["to_do"].get("rich_text", [])
                checked = block["to_do"].get("checked", False)
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    checkbox = "☑" if checked else "☐"
                    text_parts.append(f"{indent}{checkbox} {text}")

            elif block_type == "toggle":
                rich_text = block["toggle"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}▶ {text}")

            elif block_type == "code":
                rich_text = block["code"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                language = block["code"].get("language", "")
                if text.strip():
                    text_parts.append(f"{indent}```{language}\n{text}\n```")

            elif block_type == "quote":
                rich_text = block["quote"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text.strip():
                    text_parts.append(f"{indent}> {text}")

            elif block_type == "callout":
                rich_text = block["callout"].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                icon = block["callout"].get("icon", {})
                icon_str = icon.get("emoji", "💡") if icon.get("type") == "emoji" else "💡"
                if text.strip():
                    text_parts.append(f"{indent}{icon_str} {text}")

            elif block_type == "divider":
                text_parts.append(f"{indent}---")

            elif block_type == "table_of_contents":
                text_parts.append(f"{indent}[Table of Contents]")

            elif block_type == "breadcrumb":
                text_parts.append(f"{indent}[Breadcrumb]")

            elif block_type == "column_list":
                # Column lists will be handled by processing children
                pass

            elif block_type == "column":
                # Columns will be handled by processing children
                pass

            elif block_type == "table":
                # Tables are complex - just note their presence
                text_parts.append(f"{indent}[Table with {block['table'].get('table_width', 0)} columns]")

            elif block_type == "table_row":
                # Extract cell content
                cells = block.get("table_row", {}).get("cells", [])
                row_text = " | ".join([
                    "".join([t.get("plain_text", "") for t in cell])
                    for cell in cells
                ])
                if row_text.strip():
                    text_parts.append(f"{indent}| {row_text} |")

            elif block_type in ["image", "video", "file", "pdf", "bookmark", "embed"]:
                # Media blocks - just note what they are
                caption_parts = block.get(block_type, {}).get("caption", [])
                caption = "".join([t.get("plain_text", "") for t in caption_parts]) if caption_parts else ""
                if caption:
                    text_parts.append(f"{indent}[{block_type.upper()}: {caption}]")
                else:
                    text_parts.append(f"{indent}[{block_type.upper()}]")

            # Handle nested children if block has them
            if block.get("has_children", False):
                try:
                    children = await self.notion.blocks.children.list(block_id=block["id"])
                    for child_block in children.get("results", []):
                        child_text = await self.extract_block_text(child_block, indent_level + 1)
                        if child_text.strip():
                            text_parts.append(child_text)
                except Exception as e:
                    # If we can't get children, just note it
                    text_parts.append(f"{indent}  [Nested content - could not retrieve]")

        except Exception as e:
            # If block parsing fails, note it but don't crash
            text_parts.append(f"{indent}[Block type '{block_type}' - could not parse: {str(e)[:50]}]")

        return "\n".join(text_parts)

    async def extract_page_text(self, page_id: str) -> str:
        """Extract text content from a Notion page with all nested blocks"""
        try:
            blocks = await self.notion.blocks.children.list(block_id=page_id)

            text_parts = []
            for block in blocks.get("results", []):
                block_text = await self.extract_block_text(block)
                if block_text.strip():
                    text_parts.append(block_text)

            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"⚠ Warning: Could not extract page text: {e}")
            return ""

    def extract_properties(self, properties: Dict[str, Any]) -> Dict[str, str]:
        """Extract all property values from a Notion page"""
        extracted = {}

        for prop_name, prop_value in properties.items():
            prop_type = prop_value.get("type")

            if prop_type == "title":
                title_parts = prop_value.get("title", [])
                text = "".join([t.get("plain_text", "") for t in title_parts])
                if text.strip():
                    extracted[prop_name] = text

            elif prop_type == "rich_text":
                text_parts = prop_value.get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in text_parts])
                if text.strip():
                    extracted[prop_name] = text

            elif prop_type == "select":
                select_val = prop_value.get("select")
                if select_val and select_val.get("name"):
                    extracted[prop_name] = select_val["name"]

            elif prop_type == "multi_select":
                options = prop_value.get("multi_select", [])
                if options:
                    extracted[prop_name] = ", ".join([o["name"] for o in options])

            elif prop_type == "date":
                date_val = prop_value.get("date")
                if date_val and date_val.get("start"):
                    if date_val.get("end"):
                        extracted[prop_name] = f"{date_val['start']} to {date_val['end']}"
                    else:
                        extracted[prop_name] = date_val["start"]

            elif prop_type == "url":
                url_val = prop_value.get("url")
                if url_val:
                    extracted[prop_name] = url_val

            elif prop_type == "email":
                email_val = prop_value.get("email")
                if email_val:
                    extracted[prop_name] = email_val

            elif prop_type == "phone_number":
                phone_val = prop_value.get("phone_number")
                if phone_val:
                    extracted[prop_name] = phone_val

            elif prop_type == "number":
                num_val = prop_value.get("number")
                if num_val is not None:
                    extracted[prop_name] = str(num_val)

            elif prop_type == "checkbox":
                checkbox_val = prop_value.get("checkbox")
                if checkbox_val is not None:
                    extracted[prop_name] = "Yes" if checkbox_val else "No"

        return extracted

    def find_title_property(self, properties: Dict[str, Any]) -> str:
        """Dynamically find and extract the title from any title-type property"""
        # First, look for a property with type="title"
        for prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                title_parts = prop_value.get("title", [])
                text = "".join([t.get("plain_text", "") for t in title_parts])
                if text.strip():
                    return text.strip()

        # If no title found, return default
        return "Untitled"

    async def sync_page_to_memory(self, page_id: str) -> Optional[str]:
        """Sync a Notion page to Memory AI"""
        try:
            # Get page details
            page = await self.notion.pages.retrieve(page_id=page_id)

            # Extract all properties
            properties = self.extract_properties(page["properties"])

            # Dynamically find title from any title-type property
            title = self.find_title_property(page["properties"])

            # Build content from properties
            content_parts = []

            # Add properties as structured data (skip the title property)
            for prop_name, prop_value in properties.items():
                # Don't re-add the title property to content
                if page["properties"][prop_name].get("type") != "title":
                    content_parts.append(f"**{prop_name}:** {prop_value}")

            # Also get page body blocks (if any)
            page_body = await self.extract_page_text(page_id)
            if page_body.strip():
                content_parts.append(f"\n## Page Content\n\n{page_body}")

            # Combine everything
            content = "\n".join(content_parts)

            if not content.strip():
                print(f"⚠ Page '{title}' has no content, skipping...")
                return None

            # Create memory
            full_content = f"# {title}\n\n{content}"

            response = await self.memory_client.post(
                "/v1/memories",
                json={
                    "collection_id": self.collection_id,
                    "content": full_content,
                    "metadata": {
                        "source": "notion",
                        "page_id": page_id,
                        "title": title,
                        "url": page.get("url", ""),
                        "created_time": page.get("created_time", ""),
                        "last_edited_time": page.get("last_edited_time", "")
                    }
                }
            )

            memory_id = response.json()["id"]
            print(f"✓ Synced '{title}' → Memory AI (ID: {memory_id[:8]}...)")
            return memory_id

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"✗ Failed to sync page {page_id}: {e}")
            print(f"   Details: {error_details[:200]}")
            return None

    async def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search Memory AI and return results"""
        try:
            response = await self.memory_client.post(
                "/v1/search",
                json={
                    "query": query,
                    "collection_id": self.collection_id,
                    "top_k": top_k,
                    "search_type": "hybrid"
                }
            )
            return response.json()["results"]
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return []

    async def ask_question(self, question: str) -> str:
        """Ask a question about your Notion notes using RAG"""
        try:
            response = await self.memory_client.post(
                "/v1/reason",
                json={
                    "query": question,
                    "collection_id": self.collection_id,
                    "llm_provider": "gemini",  # or "openai" or "anthropic"
                    "include_sources": True
                }
            )

            # Check response status
            if response.status_code != 200:
                error_detail = response.json() if response.text else {}
                return f"❌ API Error ({response.status_code}): {error_detail}"

            result = response.json()

            # Check if answer exists
            if "answer" not in result:
                return f"❌ No answer in response. Response: {result}"

            answer = result["answer"]
            sources = result.get("sources", [])

            # Format response
            response_text = f"{answer}\n\nSources:\n"
            for i, source in enumerate(sources[:3], 1):
                title = source.get("metadata", {}).get("title", "Unknown")
                response_text += f"{i}. {title}\n"

            return response_text

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return f"❌ Error: {e}\n\nDetails:\n{error_trace}"

    async def create_notion_page_from_memory(
        self,
        database_id: str,
        memory_id: str
    ) -> Optional[str]:
        """Create a Notion page from a Memory AI memory"""
        try:
            # Get memory from Memory AI
            response = await self.memory_client.get(f"/v1/memories/{memory_id}")
            memory = response.json()

            content = memory["content"]
            metadata = memory.get("metadata", {})

            # Split title and content
            lines = content.split("\n", 1)
            title = lines[0].strip("# ").strip()
            body = lines[1] if len(lines) > 1 else ""

            # Create page in Notion
            new_page = await self.notion.pages.create(
                parent={"database_id": database_id},
                properties={
                    "Name": {"title": [{"text": {"content": title}}]}
                },
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"text": {"content": body}}]
                        }
                    }
                ]
            )

            print(f"✓ Created Notion page: {title}")
            return new_page["id"]

        except Exception as e:
            print(f"✗ Failed to create Notion page: {e}")
            return None

    async def sync_all_pages_in_database(self, database_id: str):
        """Sync all pages from a Notion database"""
        print(f"\n📚 Syncing database: {database_id}\n")

        # Query database
        response = await self.notion.databases.query(database_id=database_id)
        pages = response["results"]

        synced = 0
        for page in pages:
            page_id = page["id"]
            memory_id = await self.sync_page_to_memory(page_id)
            if memory_id:
                synced += 1

            # Rate limiting
            await asyncio.sleep(0.5)

        print(f"\n✓ Synced {synced}/{len(pages)} pages to Memory AI")

    async def close(self):
        """Close connections"""
        await self.memory_client.aclose()


# Example usage functions
async def main():
    """Main interactive menu"""
    bot = NotionMemoryBot()
    await bot.initialize()

    # Check if memories exist
    try:
        response = await bot.memory_client.get(
            f"/v1/collections/{bot.collection_id}/memories"
        )
        memory_count = len(response.json())
        print(f"\n📊 Current status: {memory_count} memories in 'Notion Notes' collection")
        if memory_count == 0:
            print("⚠️  No memories yet! Start by syncing a Notion page (option 1 or 2)")
    except Exception as e:
        print(f"⚠️  Could not check memories: {e}")

    while True:
        print("\n" + "="*60)
        print("NOTION MEMORY AI BOT - MENU")
        print("="*60)
        print("1. Sync a single Notion page")
        print("2. Sync entire Notion database")
        print("3. Search your Notion memories")
        print("4. Ask a question about your notes")
        print("5. Create Notion page from memory")
        print("6. Exit")
        print("="*60)

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            page_id = input("Enter Notion page ID: ").strip()
            await bot.sync_page_to_memory(page_id)

        elif choice == "2":
            database_id = input("Enter Notion database ID: ").strip()
            await bot.sync_all_pages_in_database(database_id)

        elif choice == "3":
            query = input("Search query: ").strip()
            results = await bot.search_memories(query)
            print(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                title = result.get("metadata", {}).get("title", "Untitled")
                score = result.get("score", 0)
                print(f"{i}. {title} (score: {score:.2f})")

        elif choice == "4":
            question = input("Your question: ").strip()
            answer = await bot.ask_question(question)
            print(f"\nAnswer:\n{answer}")

        elif choice == "5":
            database_id = input("Target Notion database ID: ").strip()
            memory_id = input("Memory ID to convert: ").strip()
            await bot.create_notion_page_from_memory(database_id, memory_id)

        elif choice == "6":
            print("\n👋 Goodbye!")
            break

        else:
            print("Invalid choice!")

    await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
