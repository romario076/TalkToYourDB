import os
import streamlit as st
from typing import Literal, Optional, TypedDict, Any
from pydantic import BaseModel, Field
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import matplotlib.pyplot as plt

MAX_SQL_ROWS = 20
FORBIDDEN_KEYWORDS = {"DELETE", "DROP", "TRUNCATE", "INSERT", "UPDATE", "ALTER", "GRANT", "REVOKE"}


class AgentState(TypedDict):
    question: str
    schema_info: str
    generated_query: str
    query_result_df: Optional[pd.DataFrame]
    query_result_str: str
    error: Optional[str]
    trials: int
    viz_code: Optional[str]
    router_decision: Optional[str]
    llm_instance: Any
    db_instance: Any


# --- MODELS & STATE ---

class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(description="True if the question is related to the database schema, False otherwise.")
    reasoning: str = Field(description="Brief explanation of why the input is or isn't relevant.")


class SQLQuery(BaseModel):
    query: str = Field(description="The syntactically correct SQL query to execute.")
    explanation: str = Field(description="A brief explanation of what this query retrieves.")


class VisualizationCode(BaseModel):
    code: str = Field(..., description="Python code using matplotlib/seaborn. Use plt.savefig('chart.png').")
    chart_description: str = Field(..., description="A sentence describing the chart.")


class RouterDecision(BaseModel):
    next_step: Literal["visualize", "summarize"]



# ---  NODE FUNCTIONS ---

def input_guardrail_node(state: AgentState):
    llm = state['llm_instance'].with_structured_output(RelevanceCheck)

    system_prompt = f"""
    You are a gatekeeper for a Medical Database assistant. 
    Your task is to determine if the user's request is related to the available database tables.

    SCHEMA:
    {state['schema_info']}

    CRITERIA:
    - If the user asks about heart rates, episodes, durations, timestamps, or medical data in these tables, it is RELEVANT.
    - If the user asks about unrelated topics (e.g., weather, jokes, general knowledge, or coding unrelated to this DB), it is NOT RELEVANT.
    """

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=state['question'])])

    if not response.is_relevant:
        return {"error": "input not relevant",
                "query_result_str": "I'm sorry, but that request is not related to the medical database I manage."}

    return {"error": None}


def generate_query_node(state: AgentState):
    # . Pass it to the instance getter
    llm = state['llm_instance'].with_structured_output(SQLQuery)

    system_prompt = f"""
    You are an expert Medical Database SQL architect.
    SCHEMA: {state['schema_info']}

    DATABASE SCHEMA CONTEXT AND RELATIONSHIPS:
    --------------------------------------------------------------------------------
    All three tables are linked by the Episode Number:
    - ICM_RSP_Columns.DEpsdNumberEpisode
    - ICM_RSP_DATA_Columns.DEpsdNumber
    - ICM_EPI_Columns.DEpsdNumber

    You MUST use JOIN operations (e.g., INNER JOIN) to retrieve data that spans across these tables.
    The primary join key is the Episode Number.

    -- TABLE ICM_RSP_Columns 
    Columns description:
    DEpsdNumberEpisode (INTEGER): Episode number identifier.
    EpsdStartSyncTS (TEXT): Episode start time.
    EpsdEndSyncTS (TEXT): Episode end time.
    StreamDataType (INTEGER): Records the history data stream type. Mappings:
        0 = StreamAnalogData
        1 = StreamEventMarkerData
        2 = StreamHeartSoundEnsembles
        3 = StreamImpedanceSamples
        4 = StreamContinuousImpedance
    TimeOfFirstSample (TEXT): Time of first sample.
    ExciterCurrent (INTEGER): Exciter current level. Mappings:
        0 = TIMM40uA 
        1 = TIMM80uA
        2 = TIMM120uA
        3 = TIMM160uA

    -- TABLE ICM_RSP_DATA_Columns 
    Columns description:
    DEpsdNumber (INTEGER): Episode number identifier.
    Obsolescence (INTEGER): Indicates whether episode has been marked obsolete or not (0=Not Obsolete, 1=Obsolete).
    MarkerSelector (INTEGER): 0 is data, 1 is marker. 

    -- TABLE ICM_EPI_Columns 
    Columns description:
    DEpsdNumber (INTEGER): Episode number identifier.
    StorEpsdType (INTEGER): Integer number between 1 and 14. Storage classification of history episode type. 
    DiagEpsdType (INTEGER): Integer number between 0 and 7. Diagnostic classification of history episode type.
    Specialness (INTEGER): Indicates that this is a special event (1=True) or not special (0=False). 
    VT_EpisodeDuration (REAL): The length of the episode in seconds.
    VT_AverageHeartRate (REAL): The average heart rate over the episode duration.
    VT_MaxHeartRate (REAL): The highest rate value reached during the episode.

    CRITICAL RULES:
    1. Output ONLY the SQL logic required.
    2. TIME DIFFERENCES (SQLite): To get the difference in seconds between two timestamps, 
       use: (strftime('%s', end_col) - strftime('%s', start_col)).
    3. Obsolescence: 0=Not Obsolete, 1=Obsolete.
    4. For "distribution" or "counts" requests, ensure you GROUP BY the relevant column and COUNT(*) so the data is ready for a bar or pie chart.
    """
    messages = [SystemMessage(content=system_prompt)]
    if state.get("error"):
        messages.append(HumanMessage(content=f"Previous query failed: {state['error']}. Fix it."))
        messages.append(AIMessage(content=state['generated_query']))
    else:
        messages.append(HumanMessage(content=state['question']))

    response = llm.invoke(messages)
    return {"generated_query": response.query, "trials": state["trials"] + 1, "error": None}


def execute_query_node(state: AgentState):
    query = state["generated_query"]
    db = state['db_instance']

    # Simple syntax check
    if any(k in query.upper() for k in FORBIDDEN_KEYWORDS):
        return {"error": "Forbidden keyword detected.", "query_result_str": "Error"}

    try:
        from sqlalchemy import text
        with db._engine.connect() as connection:
            cursor = connection.execute(text(query))
            rows = cursor.fetchmany(MAX_SQL_ROWS)
            columns = list(cursor.keys())
            result_df = pd.DataFrame(rows, columns=columns)
            return {
                "query_result_df": result_df,
                "query_result_str": result_df.to_string(index=False),
                "error": None
            }
    except Exception as e:
        return {"query_result_str": "Error",
                "query_result_df": None,
                "error": str(e)
                }


def router_node(state: AgentState):

    # Pass to instance getter
    llm = state['llm_instance'].with_structured_output(RouterDecision)

    # prompt = f"User Question: '{state['question']}'. Return 'visualize' if asked for a chart or plot, else 'summarize'."

    prompt = f"""
        Analyze the User Question: "{state['question']}"

        Determine if the user wants a visual representation (chart, plot, distribution, histogram, scatter, etc.) 
        or a text-based answer/calculation.

        - If the user mentions "distribution", "plot", "chart", "graph", "histogram", "scatter", "visualize", return 'visualize'.
        - If the user asks for a list, a count, a single number, calculate, summary or something similar, return 'summarize'.
        """

    decision = llm.invoke(prompt)
    return {"router_decision": decision.next_step}


def generate_chart_node(state: AgentState):
    df = state['query_result_df']
    if df is None or df.empty:
        return {"viz_code": None}

    # Pass to instance getter
    llm = state['llm_instance'].with_structured_output(VisualizationCode)

    data_context = f"Columns: {list(df.columns)}\nTypes: {df.dtypes}"
    sample = df.head(3).to_string()

    # prompt = f"""
    # User Question: "{state['question']}"
    # Data: {data_context}
    # Sample: {sample}
    # Generate Python code using matplotlib/seaborn.
    # IMPORTANT: Do NOT use plt.show(). Use plt.savefig('chart.png').
    # """

    prompt = f"""
        You are a Python Data Visualization expert. The user wants a visualization of the data in the pandas DataFrame `df`.

        User Question: "{state['question']}"

        DataFrame Schema:
        {data_context}

        DataFrame Sample (First 3 rows ONLY - usage guidance below):
        {sample}

        **CRITICAL CODE INSTRUCTIONS**:
        1. **USE ALL DATA**: The `df` variable contains the FULL dataset (more than the 3 sample rows shown above). Do NOT use `head()` or slice the dataframe in your code. Plot every row.
        2. **HANDLE DATES**: SQLite returns dates as strings. You MUST convert any date/time columns used in the plot to datetime objects first using `pd.to_datetime()`. 
           - Example: `df['MyDateCol'] = pd.to_datetime(df['MyDateCol'])`
        3. **VISUALS**: Use `plt.figure(figsize=(10, 6))` to ensure the dates don't overlap. Use `plt.xticks(rotation=45)` if plotting dates on X-axis. Add titles, labels, grid.
        4. **OUTPUT**: Write Python code to generate the plot and save it using `plt.savefig('chart.png')`. Do NOT use `plt.show()`.
        5. **AUTOMATIC PLOT SELECTION**: 
           - If the data has 2 columns and one is numeric, use a Bar Chart or Histogram.
           - If there are 2 numeric columns, use a Scatter Plot.
           - If there is a Date column, use a Line Plot.
        6. **LABELING**: Always include `plt.title()`, `plt.xlabel()`, and `plt.ylabel()`.
        """
    response = llm.invoke(prompt)
    return {"viz_code": response.code}


def formulate_answer_node(state: AgentState):
    prompt = f"Question: {state['question']}\nResult: {state['query_result_str']}\nAnswer professionally."
    response = state['llm_instance'].invoke(prompt)
    return {"query_result_str": response.content}


def exec_python_visualization_code(state: AgentState):
    df = state['query_result_df']
    code = state['viz_code']

    # Clean up old chart
    if os.path.exists("chart.png"):
        os.remove("chart.png")

    if df is not None and not df.empty and code:
        try:
            plt.clf()
            plt.close('all')
            plt.switch_backend('Agg')
            plt.figure()
            local_scope = {"df": df.copy(), "plt": plt, "pd": pd}
            exec(code, {}, local_scope)
            if os.path.exists("chart.png"):
                return {"query_result_str": "Chart generated."}
        except Exception as e:
            return {"error": f"Viz Error: {e}"}
    return {"error": "No data or code."}
