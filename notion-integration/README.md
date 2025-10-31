# Notion ↔ Memory AI Integration 🧠

**Direct integration between Notion and your Memory AI API**

Automatically sync your Notion pages to Memory AI for intelligent search, RAG-powered Q&A, and advanced memory features.

---

## ✨ Features

- ✅ **Sync Notion pages** to Memory AI automatically
- ✅ **Sync entire databases** with one command
- ✅ **Search your notes** using hybrid vector + BM25 search
- ✅ **Ask questions** - RAG-powered answers from your Notion notes
- ✅ **Create pages** from Memory AI memories
- ✅ **Real-time sync** - keep Notion and Memory AI in sync
- ✅ **Supports all block types** - paragraphs, headings, lists, code, etc.

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd "/Users/manavpatel/Documents/API Memory/notion-integration"
pip3 install -r requirements.txt
```

### Step 2: Get Notion Integration Token

1. Go to: https://www.notion.so/my-integrations
2. Click "**+ New integration**"
3. Name: "**Memory AI Bot**"
4. Capabilities: Check **Read content**, **Update content**, **Insert content**
5. Click "**Submit**"
6. **Copy the Integration Token** (starts with `secret_...`)

### Step 3: Connect Integration to Your Notion Pages

**IMPORTANT**: You must share your Notion pages with the integration:

1. Open your Notion page or database
2. Click **"..."** (3 dots) in top right
3. Click "**Add connections**"
4. Select "**Memory AI Bot**" (or whatever you named it)
5. Click "**Confirm**"

### Step 4: Configure Environment

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Your Notion integration token
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Your Memory AI API key (get from Memory AI API)
MEMORY_AI_API_KEY=mem_sk_your_api_key

# Memory AI API URL
MEMORY_AI_BASE_URL=http://localhost:8000
```

### Step 5: Get Your Memory AI API Key

```bash
# Start Memory AI API first
cd "/Users/manavpatel/Documents/API Memory"
# (make sure API is running on port 8000)

# Register and get API key
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "password": "secure_password",
    "full_name": "Your Name"
  }'

# Copy the "access_token" from response
```

### Step 6: Run the Bot

```bash
python3.11 notion_memory_bot.py
```

---

## 📖 Usage Examples

### Example 1: Sync a Single Page

```
Menu choice: 1
Enter Notion page ID: abc123def456...
✓ Synced 'My Work Notes' → Memory AI (ID: mem_xyz...)
```

**How to get page ID:**
- Open Notion page
- URL: `https://notion.so/My-Work-Notes-abc123def456...`
- Page ID = `abc123def456...`

### Example 2: Sync Entire Database

```
Menu choice: 2
Enter Notion database ID: db_789xyz...
✓ Synced 15/15 pages to Memory AI
```

**How to get database ID:**
- Open database in Notion
- URL: `https://notion.so/db_789xyz...`
- Database ID = `db_789xyz...`

### Example 3: Search Your Notes

```
Menu choice: 3
Search query: project deadlines
Found 5 results:
1. Q4 Planning Meeting (score: 0.92)
2. Project Timeline (score: 0.87)
3. Deadlines 2025 (score: 0.85)
```

### Example 4: Ask Questions

```
Menu choice: 4
Your question: What are my main priorities this quarter?

Answer:
Based on your notes, your main Q4 priorities are:
1. Launch new product feature by Oct 15
2. Complete customer onboarding improvements
3. Hire 2 engineers for backend team

Sources:
1. Q4 Planning Meeting
2. Team OKRs 2025
3. Product Roadmap
```

---

## 🎯 Real-World Use Cases

### 1. **Personal Knowledge Base**
Sync all your personal notes to Memory AI, then ask questions like:
- "What did I learn about Python decorators?"
- "Show me all notes about machine learning"

### 2. **Meeting Notes**
Sync meeting notes database, then:
- "What was discussed in the last standup?"
- "Find all action items from client meetings"

### 3. **Project Documentation**
Keep project docs in Notion, use Memory AI for:
- "How does the authentication system work?"
- "Show me all API endpoints"

### 4. **Learning Journal**
Track what you learn, then:
- "Summarize what I've learned this month"
- "Find resources about React hooks"

### 5. **Task Management**
Sync tasks/todos:
- "What are my high priority tasks?"
- "Show me overdue items"

---

## 🔧 Advanced Features

### Auto-Sync with Webhooks (Coming Soon)

For real-time sync, you can set up Notion webhooks:

```python
# webhook_server.py - coming soon
# Listens for Notion changes and auto-syncs
```

### Bidirectional Sync

Create Notion pages from Memory AI:

```python
bot = NotionMemoryBot()
await bot.initialize()

# Create page from memory
await bot.create_notion_page_from_memory(
    database_id="your_db_id",
    memory_id="mem_xyz..."
)
```

### Custom Metadata

Add custom tags and metadata:

```python
# Customize sync behavior
await bot.sync_page_to_memory(
    page_id="page_123",
    extra_metadata={
        "tags": ["important", "work"],
        "category": "engineering"
    }
)
```

---

## 🐛 Troubleshooting

### Issue: "Unauthorized" Error

**Problem**: Integration doesn't have access to page

**Solution**:
1. Open the Notion page
2. Click "..." → "Add connections"
3. Select your integration
4. Click "Confirm"

### Issue: "Page has no content"

**Problem**: Page is empty or has unsupported blocks

**Solution**:
- Add text content to the page
- Supported blocks: paragraphs, headings, lists, code, quotes
- Not yet supported: images, embeds, databases

### Issue: "Connection refused" to Memory AI

**Problem**: Memory AI API is not running

**Solution**:
```bash
cd "/Users/manavpatel/Documents/API Memory"
# Start your Memory AI API
```

### Issue: "Invalid API key"

**Problem**: Wrong or expired API key

**Solution**:
- Register again to get new API key
- Update `.env` file with new key

---

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌───────────────┐
│   Notion    │ ──────> │  Integration Bot │ ──────> │  Memory AI    │
│   Pages     │ <────── │  (This Script)   │ <────── │     API       │
└─────────────┘         └──────────────────┘         └───────────────┘
                                │
                                │
                        ┌───────v────────┐
                        │  Vector Search │
                        │  RL Training   │
                        │  Knowledge     │
                        │  Graphs        │
                        └────────────────┘
```

**How it works:**

1. **Read** Notion pages via Notion API
2. **Extract** text from all block types
3. **Send** to Memory AI API
4. **Store** in vector database with metadata
5. **Search** using hybrid search (vector + BM25)
6. **Answer** questions using RAG

---

## 📊 Supported Notion Blocks

| Block Type | Supported | Notes |
|-----------|-----------|-------|
| Paragraph | ✅ | Full support |
| Heading 1/2/3 | ✅ | Preserved as markdown |
| Bulleted list | ✅ | Converted to `•` |
| Numbered list | ✅ | Converted to `-` |
| To-do | ✅ | Shows ☐/☑ |
| Code | ✅ | With language tag |
| Quote | ✅ | As `>` markdown |
| Callout | ⏳ | Coming soon |
| Table | ⏳ | Coming soon |
| Image | ⏳ | Coming soon |
| Embed | ⏳ | Coming soon |

---

## 🚀 What's Next?

After syncing your Notion to Memory AI, you can:

1. **Train RL agents** - Optimize what to remember
2. **Build knowledge graphs** - See connections between notes
3. **Enable temporal tracking** - Track how ideas evolve
4. **Create procedures** - Automate workflows
5. **Use world models** - Simulate before acting

---

## 📝 API Reference

### NotionMemoryBot Class

```python
bot = NotionMemoryBot()
await bot.initialize()

# Sync single page
await bot.sync_page_to_memory(page_id="...")

# Sync database
await bot.sync_all_pages_in_database(database_id="...")

# Search
results = await bot.search_memories(query="...", top_k=5)

# Ask question
answer = await bot.ask_question(question="...")

# Create Notion page
page_id = await bot.create_notion_page_from_memory(
    database_id="...",
    memory_id="..."
)
```

---

## 💡 Tips

1. **Start small** - Sync a few pages first to test
2. **Use databases** - Easier to manage than individual pages
3. **Tag your notes** - Use Notion tags for better organization
4. **Regular sync** - Run sync daily to keep up-to-date
5. **Ask specific questions** - More specific = better answers

---

## 🤝 Contributing

Want to improve this integration?

- Add support for more block types
- Implement webhooks for real-time sync
- Add bidirectional sync
- Create Notion database from Memory AI

---

## 📄 License

MIT License - Free to use and modify

---

## 🔗 Links

- **Memory AI API**: http://localhost:8000/docs
- **Notion API Docs**: https://developers.notion.com
- **Integration Guide**: https://www.notion.so/my-integrations

---

**Built with ❤️ for seamless Notion ↔ Memory AI integration**
