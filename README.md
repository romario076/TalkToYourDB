# 🏥 SQL Data Agent (Streamlit + LLM)

An interactive **Streamlit application** that lets you ask **natural-language questions** about medical episode data and instantly receive **SQL-backed answers, tables, and visualizations** — without writing SQL.

The app translates user questions into safe SQL queries, executes them against a structured database, and optionally generates charts using AI.

---

## ✨ Features

- 💬 Ask questions in **natural language**
- 🧠 Automatic **SQL generation** using LLMs (OpenAI GPT-4o or AWS Bedrock Claude)
- 🔒 **Read-only SQL execution** with safety checks
- 📊 Automatic **data visualizations** (bar, line, scatter, histograms)
- 🔁 Query retry logic with error correction
- 🧭 Intent routing: **summary vs visualization**
- 🗄️ SQLite database with realistic medical episode data
- 🧩 Modular **LangGraph** workflow

---

## 🧱 Data Background (Simplified)

This application is built on top of **DET-like medical reports**, derived from OUD files:

1. OUD files are parsed using a **DET parser**
2. Parsed JSON documents are **unnested into relational tables**
3. Flat tables are used to generate **DET-like reports**
4. A **synthetic database** based on these reports powers this application

---

## 🏗️ Architecture Overview

**High-level flow:**

User Question
↓
LLM → SQL Generation
↓
SQL Validation & Execution
↓
Intent Routing
↓
Text Answer OR Chart Generation


**Core components:**

- **Streamlit** – UI and user interaction
- **LangChain / LangGraph** – Agent orchestration
- **LLMs** – SQL generation, routing, and explanations
- **SQLite + SQLAlchemy** – Database and query execution
- **Matplotlib / Pandas** – Visualization and data handling

---

## 🔁 Agent Workflow (LangGraph)

1. **Generate SQL**
   - LLM converts user question into SQL using schema context

2. **Execute SQL**
   - Query is validated (read-only)
   - Executed with row limits

3. **Retry on Error**
   - SQL is regenerated if execution fails (up to 3 attempts)

4. **Intent Router**
   - Determines whether user wants:
     - 📊 visualization
     - 📝 text summary

5. **Output**
   - Chart (saved as image) **or**
   - Professional text answer

---

## 📊 Supported Visualizations

The app automatically selects chart types based on query results:

- Bar charts
- Line charts (time series)
- Scatter plots
- Histograms

All charts are generated using **matplotlib** and rendered in Streamlit.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key

# Optional: AWS Bedrock
AWS_DEFAULT_REGION=us-east-1
CLAUDE_MODEL_ID=anthropic.claude-3-5-sonnet
```

▶️ Run the Application

```
pip install -r requirements.txt
streamlit run app.py
```
