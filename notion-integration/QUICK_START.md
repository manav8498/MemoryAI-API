# Quick Start - Sync Your Notion Database

## Step 1: Get Your Database ID

1. Open your database in Notion (table, board, gallery, etc.)
2. Click **"Share"** → **"Copy link"**
3. URL looks like: `https://www.notion.so/workspace/Database-Name-a1b2c3d4...?v=...`
4. Copy the **ID part** (the long string after the name, before `?v=`)

**Examples:**
```
URL: https://notion.so/My-Notes-abc123def456?v=xyz
Database ID: abc123def456

URL: https://notion.so/workspace/Job-Applications-123abc456def
Database ID: 123abc456def
```

---

## Step 2: Share Database with Integration

1. In your database, click **"..."** (3 dots in top right)
2. Click **"Add connections"**
3. Select your integration name
4. Click **"Confirm"**

---

## Step 3: Sync Your Database

```bash
cd "/Users/manavpatel/Documents/API Memory/notion-integration"
python3.11 sync_database.py YOUR_DATABASE_ID
```

**Example:**
```bash
python3.11 sync_database.py abc123def456
```

This will automatically sync ALL pages in your database!

---

## Step 4: Ask Questions

After syncing, ask questions about your notes:

```bash
python3.11 ask.py "What are my priorities this month?"
python3.11 ask.py "Show me all meeting notes"
python3.11 ask.py "What did I learn about Python?"
```

---

## Full Example Workflow

```bash
# 1. Go to directory
cd "/Users/manavpatel/Documents/API Memory/notion-integration"

# 2. Sync your database
python3.11 sync_database.py abc123def456

# 3. Ask questions
python3.11 ask.py "What are my top priorities?"
python3.11 ask.py "Summarize my meeting notes"
python3.11 ask.py "What tasks are incomplete?"
```

---

## Tips

- **Sync once** - Your database gets synced completely
- **Re-sync anytime** - Run sync again to update with latest changes
- **Multiple databases** - Sync different databases by running the command with different IDs
- **Ask anything** - The AI understands natural language questions

---

## Troubleshooting

**"Unauthorized" error:**
- Make sure you shared the database with your integration (Step 2)

**"Page has no content":**
- Some pages in database might be empty - that's OK, they're skipped

**Connection error:**
- Make sure Memory AI API is running: check `http://localhost:8000/docs`

---

## Need the Full Menu?

For more options (search, create pages, etc.):
```bash
python3.11 notion_memory_bot.py
```

---

**That's it! Your Notion notes are now searchable with AI!** 🎉
