
import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
from langchain_community.utilities import SQLDatabase
from langgraph.graph import END, StateGraph, START

from tools.nodes import AgentState
from tools.setup_database import setup_database
from tools.nodes import (input_guardrail_node, generate_query_node, execute_query_node,
                         router_node, generate_chart_node, exec_python_visualization_code, formulate_answer_node)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'),override=True)

# --- 1. CONFIGURATION ---
DB_FILE = "medical_data.db"
DB_URI = f"sqlite:///{DB_FILE}"
LLM_MODEL = "gpt-4o"
MAX_SQL_ROWS = 20
FORBIDDEN_KEYWORDS = {"DELETE", "DROP", "TRUNCATE", "INSERT", "UPDATE", "ALTER", "GRANT", "REVOKE"}




@st.cache_resource
def get_db_instance():
    setup_database(DB_FILE=DB_FILE)
    return SQLDatabase.from_uri(DB_URI)


@st.cache_resource
def get_llm_instance(provider: str):
    print(f"--- Initializing LLM: {provider} ---")

    if provider == "AWS Bedrock (Claude 3.5)":
        return ChatBedrockConverse(
            model_id=os.environ['CLAUDE_MODEL_ID'],
            region_name=os.environ['AWS_DEFAULT_REGION'],
            provider='anthropic'
        )
    else:
        # Default to OpenAI
        return ChatOpenAI(model=LLM_MODEL, temperature=0)



# --- GRAPH BUILDER ---
def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guardrail_node", input_guardrail_node)

    workflow.add_node("generate_query_node", generate_query_node)
    workflow.add_node("execute_query_node", execute_query_node)
    workflow.add_node("router_node", router_node)
    workflow.add_node("generate_chart_node", generate_chart_node)
    workflow.add_node("exec_python_visualization_code", exec_python_visualization_code)
    workflow.add_node("formulate_answer_node", formulate_answer_node)

    workflow.add_edge(START, "input_guardrail_node")

    def check_relevance(state):
        if state.get("error") == "input not relevant":
            return END
        return "generate_query_node"

    workflow.add_conditional_edges(
        "input_guardrail_node",
        check_relevance,
        {END: END, "generate_query_node": "generate_query_node"}
    )

    workflow.add_edge("generate_query_node", "execute_query_node")

    def route_after_exec(state):
        if state["trials"] > 3: return END
        if state["error"]: return "generate_query_node"
        return "router_node"

    workflow.add_conditional_edges("execute_query_node", route_after_exec,
                                   {"generate_query_node": "generate_query_node", "router_node": "router_node",
                                    "__end__": END})

    def route_intent(state):
        return "generate_chart_node" if state["router_decision"] == "visualize" else "formulate_answer_node"

    workflow.add_conditional_edges("router_node", route_intent,
                                   {"generate_chart_node": "generate_chart_node",
                                    "formulate_answer_node": "formulate_answer_node"})

    workflow.add_edge("generate_chart_node", "exec_python_visualization_code")
    workflow.add_edge("exec_python_visualization_code", END)
    workflow.add_edge("formulate_answer_node", END)

    return workflow.compile()


# --- 6. STREAMLIT UI ---

st.set_page_config(page_title="SQL Data Agent", layout="wide", page_icon="🏥")

st.title("🏥 SQL Data Agent")
st.markdown("Ask questions about your database. The agent can query data and generate visualizations.")

# --- SIDEBAR: Database Preview ---
with st.sidebar:
    st.header("⚙️ Settings")

    # --- LLM SWITCHER ---
    llm_option = st.radio(
        "Select LLM Provider:",
        ("OpenAI (GPT-4o)", "AWS Bedrock (Claude 3.5)"),
        index=1
    )
    # Store selection in session state so nodes can access it
    st.session_state["selected_provider"] = llm_option

    st.divider()

    st.header("📂 Database Preview")
    db = get_db_instance()

    table_names = db.get_usable_table_names()

    selected_table = st.selectbox("Select Table to View", table_names)

    if selected_table:
        with st.expander(f"Preview: {selected_table}", expanded=True):
            try:
                # Use SQLalchemy connection for preview
                with db._engine.connect() as conn:
                    # Use double quotes for table name to be safe
                    df_preview = pd.read_sql(f'SELECT * FROM "{selected_table}" LIMIT 5', conn)
                    st.dataframe(df_preview, use_container_width=True)
                    st.caption(f"Showing first 5 rows of {selected_table}")
            except Exception as e:
                st.error(f"Could not load table: {e}")

    st.divider()
    st.info("💡 **Tip:** Try asking for charts like 'Show me a bar chart of average heart rate'.")

# --- MAIN CHAT INTERFACE ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])
        if "dataframe" in message:
            with st.expander("View Data"):
                st.dataframe(message["dataframe"])

# User Input
if prompt := st.chat_input("Ask a question about the medical data..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize Graph Logic
    app = build_workflow()
    schema_context = db.get_table_info()

    initial_state: AgentState = {
        "question": prompt,
        "schema_info": schema_context,
        "trials": 0,
        "error": None,
        "generated_query": "",
        "query_result_str": "",
        "query_result_df": None,
        "viz_code": None,
        "router_decision": None,
        "llm_instance": get_llm_instance(
            provider=st.session_state.get("selected_provider", "OpenAI (GPT-4o)")
        ),
        "db_instance": get_db_instance(),
    }

    with st.chat_message("assistant"):
        status_container = st.status("Thinking...", expanded=True)

        try:
            # Stream events or just invoke (using invoke for simplicity here)
            status_container.write("🔍 Analyzing Schema...")
            final_state = app.invoke(initial_state)

            response_content = ""
            response_image = None
            response_df = final_state.get('query_result_df')
            response_sql = final_state.get('generated_query')  # Capture SQL

            # Handle Output
            if final_state.get("router_decision") == "visualize" and os.path.exists("chart.png"):
                status_container.write("🎨 Router decision: visualize. Generating Chart...")
                status_container.update(label="Done!", state="complete", expanded=False)

                response_content = "Here is the visualization based on your request."
                st.markdown(response_content)
                st.image("chart.png")
                response_image = "chart.png"  # Store path for history

            else:
                status_container.write("Router decision: summarize")
                status_container.write("📝 Formulating Answer...")
                status_container.update(label="Done!", state="complete", expanded=False)

                response_content = final_state.get('query_result_str', "I couldn't process that.")
                st.markdown(response_content)

            # Show Data Expander if dataframe exists
            if response_sql:
                with st.expander("🛠️ View SQL Query"):
                    st.code(response_sql, language="sql")

                # Show Data Expander
            if response_df is not None and not response_df.empty:
                with st.expander("View Source Data"):
                    st.dataframe(response_df)

            # Save to history
            msg_data = {"role": "assistant", "content": response_content}
            if response_image:
                msg_data["image"] = response_image
            if response_df is not None:
                msg_data["dataframe"] = response_df

            st.session_state.messages.append(msg_data)

        except Exception as e:
            status_container.update(label="Error", state="error")
            st.error(f"An error occurred: {e}")


def render_help_page():
    st.title("ℹ️ How This Medical Data Assistant Works")

    st.markdown("""
    This application allows you to ask **natural-language questions** about medical episode data
    and instantly receive **clear answers, tables, and charts** — without needing SQL or technical knowledge.

    ### How the data is processed:
    1. **Parse OUD Files** – Raw OUD files are processed using the DET parser.
    2. **Flatten JSON** – The parsed JSON documents are unnested into flat relational tables.
    3. **Generate DET-like Reports** – Using the flattened data and a SQL engine, standardized reports are created.
    4. **This Application** – Based on these reports was created synthetic database which is used for this application
    """)

    st.divider()

    st.subheader("🔁 End-to-End Process Overview")

    st.code("""
┌────────────────────┐
│    User Input      │
│ (Natural Question) │
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────┐
│  Web Interface           │
│  (Streamlit Application) │
└─────────┬────────────────┘
          │
          ▼
┌────────────────────────────────────── ┐
│  Relevance Check (AI Guardrail)       │
│-------------------------------------- │
│ • Is the question about medical data? │
│ • Uses DB schema as context           │
│                                       │
│ YES ────────────────▶ Continue        │
│ NO  ────────────────▶ Polite rejection│
└─────────┬──────────────────────────── ┘
          │
          ▼
┌──────────────────────────────────────┐
│ AI SQL Generator                     │
│--------------------------------------│
│ • Converts question to SQL           │
│ • Knows table relationships          │
│ • Read-only by design                │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│ SQL Execution & Validation           │
│--------------------------------------│
│ • Blocks unsafe SQL                  │
│ • Executes query                     │
│ • Limits returned rows               │
│                                      │
│ SUCCESS ───────────────▶ Continue    │
│ ERROR   ───────────────▶ Retry Logic │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│ Error Handling & Retry Controller    │
│--------------------------------------│
│ • Tracks number of attempts (Trials) │
│ • Auto-fixes SQL errors              │
│                                      │
│ Trials ≤ 3 ─────▶ Regenerate SQL     │
│ Trials > 3 ─────▶ Stop & fail safely │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│ Router Decision                      │
│--------------------------------------│
│ Does the user want:                  │
│                                      │
│ 📊 Visualization? ──▶ Chart Flow     │
│ 📝 Text / Numbers? ─▶ Summary Flow   │
└─────────┬────────────────────────────┘
          │
          ├─────────────────────────────┐
          ▼                             ▼
┌───────────────────────┐   ┌────────────────────────┐
│ Visualization Engine  │   │ Professional Answer    │
│-----------------------│   │ Generator              │
│ • Auto chart type     │   │ • Clear explanation    │
│ • Dates handled       │   │ • Business language    │
│ • Saved as image      │   └─────────┬──────────────┘
└─────────┬─────────────┘              │
          │                            │
          └───────────────┬────────────┘
                          ▼
┌──────────────────────────────────────┐
│ Final Business Output                │
│--------------------------------------│
│ • Text answer                        │
│ • Chart (if requested)               │
│ • Source table                       │
│ • SQL shown for transparency         │
└──────────────────────────────────────┘

    """, language="text")

    st.divider()

    st.subheader("🧭 Step-by-Step Explanation")

    st.markdown("""
    ### **Step 1 — Ask a Question**
    You type a question in plain English.

    **Example:**  
    *“Show me a chart of average heart rate by episode.”*
    """)

    st.markdown("""
    ### **Step 2 — Relevance Check (Safety Gate)**
    Before doing anything, the system checks:

    - ❓ Is the question about **medical episodes** and related data?
    - ❌ Or is it unrelated (weather, jokes, general coding, etc.)?

    **Outcome:**
    - ✅ If relevant → the system continues
    - ❌ If not relevant → you receive a polite message explaining it’s outside scope

    This ensures answers stay accurate and meaningful.
    """)

    st.markdown("""
    ### **Step 3 — AI Generates a Safe SQL Query**
    The AI automatically:

    - Understands the medical database structure
    - Correctly links data using the **Episode Number**
    - Generates **read-only SQL queries**
    - Blocks unsafe operations such as:
        - DELETE
        - UPDATE
        - DROP
        - ALTER

    Your data is **never modified**.
    """)

    st.markdown("""
    ### **Step 4 — Query Is Executed**
    The system then:

    - Executes the query securely
    - Returns only a **limited number of rows**
    - Converts the results into a clean, readable table
    """)

    st.markdown("""
    ### **Step 5 — Intent Detection (Router Decision)**
    The AI decides what kind of response is best:

    - 📊 **Visualization** (chart, distribution, trend)
    - 📝 **Text / Table summary** (counts, lists, explanations)

    You don’t need to specify — the system decides automatically.
    """)

    st.markdown("""
    ### **Step 6A — If a Visualization Is Needed**
    When a chart is appropriate:

    - The AI generates Python chart code
    - The chart type is chosen automatically
    - The chart is created and displayed in the app
    """)

    st.markdown("""
    ### **Step 6B — If a Summary Is Needed**
    When text is more appropriate:

    - The AI writes a **professional explanation**
    - The answer is based only on real database results
    - No assumptions or fabricated data
    """)

    st.markdown("""
    ### **Step 7 — Final Answer Is Shown**
    You receive:

    - ✅ A clear text explanation
    - 📊 A chart (if requested)
    - 📋 A table with the source data
    - 🛠 The SQL query (optional, for transparency)
    """)

    st.divider()

    st.subheader("💼 Why This Is Valuable ")

    st.markdown("""
    - No SQL or technical skills required  
    - Fast insights from complex medical data  
    - Visual trends instantly available
    - Safe, read-only access to data  
    - Transparent and explainable logic  
    - Scalable to real production databases
    """)

    st.divider()

    st.markdown("""
    ## 🚀 Planned Improvement: 3-Pass SQL Generation

    As a future enhancement, the application can use a **three-pass SQL generation workflow**
    to significantly improve query accuracy when working with complex databases.

    Instead of generating SQL in a single step, the system will follow these stages:

    ### 1️⃣ PLAN
    The language model analyzes the user question and creates a structured plan:
    - which tables are required
    - which columns are needed
    - how tables should be joined

    ### 2️⃣ DRAFT SQL
    Based on the plan, the model generates an initial (draft) SQL query.

    ### 3️⃣ VALIDATION & FINAL SQL
    The draft query is validated to ensure:
    - all referenced tables exist
    - all columns match the database schema
    - join keys are valid

    Using this validation feedback, the model produces a corrected **final SQL query**
    that is ready for execution.

    ---

    ### ✅ Why Use This Approach?
    - Reduces SQL generation errors by approximately **50–70%**
    - Improves reliability for **large and complex schemas**
    - Makes SQL generation easier to debug and extend

    ### ⚠ When Is It Needed?
    - **Databases with 10+ tables** → strongly recommended  
    - **Smaller schemas (5–6 tables)** → single-pass SQL may be sufficient
    """)

    st.divider()

    st.subheader("🧾 Executive Summary")

    st.info(
        "This system enables business users to ask natural-language questions about medical episode data "
        "and receive instant, safe, and explainable insights with tables and charts — without any technical expertise."
    )


with st.expander("ℹ️ How it works", expanded=False):
    render_help_page()