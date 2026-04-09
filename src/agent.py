import os
import json
import time
import logging
import re
from typing import Any
from dotenv import load_dotenv
import google.generativeai as genai

from src.bigquery_client import run_query
from src.schema_retriever import retrieve as schema_retrieve
from src.text_retriever import retrieve as text_retrieve

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
DATASET    = os.getenv("BIGQUERY_DATASET")

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are PitWall, an expert F1 analytics assistant with deep knowledge
of Formula 1 history, statistics, driver careers, team performance, and the business
and glamour side of the sport.

You have access to three tools:
1. get_schema_context     — retrieves relevant BigQuery table schemas for SQL generation
2. get_narrative_context  — retrieves relevant text from driver/constructor Wikipedia bios
3. run_sql                — executes a SQL query on BigQuery and returns results

BigQuery project : {PROJECT_ID}
BigQuery dataset : {DATASET}

STRICT RULES — follow every one of these without exception:
- Always call get_schema_context before writing any SQL — never guess column names
- After calling get_schema_context, you MUST always follow up with run_sql — never stop after schema retrieval alone
- For questions about driver personalities, careers, history or "why" questions, call get_narrative_context
- For hybrid questions (stats + narrative), call BOTH get_narrative_context AND get_schema_context, then run_sql
- After retrieving narrative context for a hybrid question, you MUST still run SQL for the stats — never stop after narrative alone
- Only give a final text answer when you have BOTH retrieved context AND executed SQL and have real results in hand
- Always use fully qualified table names: `{PROJECT_ID}.{DATASET}.table_name`
- Handle NULL values gracefully in SQL (use COALESCE or IS NOT NULL filters)
- The glamour_index table contains driver brand value, social following, luxury brand partnerships and glamour score — always use it for commercial, popularity, fame, or overhyped driver questions
- When results come back, synthesise a clear, engaging English answer with a formatted table where appropriate
- Cite which tables and sources you used at the end of your answer
- If SQL returns no results, explain why and try an alternative query
- Keep answers concise but complete — lead with the key insight
"""

# ── Tool definitions (genai.protos format for v0.7.x) ────────────────
# To add a new tool in future:
#   1. Add a FunctionDeclaration here
#   2. Add a handler in _execute_tool()
#   Agent learns to use it automatically from the description alone.

TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_schema_context",
                description=(
                    "Retrieves the most relevant BigQuery table schemas for a given question. "
                    "Always call this before writing SQL to get accurate table names and column names. "
                    "Returns table descriptions, column names, and example questions each table can answer."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "question": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The user's question or the SQL you are planning to write",
                        )
                    },
                    required=["question"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="get_narrative_context",
                description=(
                    "Retrieves relevant text passages from F1 driver and constructor Wikipedia biographies. "
                    "Use this for questions about driver personalities, career histories, rivalries, "
                    "team culture, or any 'why' and 'who' questions that SQL cannot answer."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "question": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The question to search narrative context for",
                        )
                    },
                    required=["question"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="run_sql",
                description=(
                    "Executes a SQL query on BigQuery and returns the results. "
                    "Only call this after retrieving schema context to ensure column names are correct. "
                    "Returns a list of result rows."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "query": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Valid BigQuery SQL query to execute",
                        )
                    },
                    required=["query"],
                ),
            ),
        ]
    )
]


# ── Tool executor ────────────────────────────────────────────────────
# Maps tool names → Python functions.
# To add a new tool: add elif block here + FunctionDeclaration above.

def _execute_tool(tool_name: str, tool_args: dict) -> Any:
    """Execute a tool by name and return its result."""

    if tool_name == "get_schema_context":
        results   = schema_retrieve(tool_args["question"], top_k=3)
        formatted = []
        for r in results:
            formatted.append(
                f"Table: {r['table']}\n"
                f"Description: {r['description']}\n"
                f"Columns: {r['columns']}\n"
                f"Relevance: {r['relevance_score']}"
            )
        return "\n\n".join(formatted)

    elif tool_name == "get_narrative_context":
        results   = text_retrieve(tool_args["question"], top_k=3)
        formatted = []
        for r in results:
            formatted.append(
                f"Source: {r['source']}\n"
                f"Relevance: {r['relevance_score']}\n"
                f"Text: {r['text'][:500]}"
            )
        return "\n\n".join(formatted)

    elif tool_name == "run_sql":
        try:
            rows = run_query(tool_args["query"])
            if not rows:
                return "Query executed successfully but returned no results."
            rows = rows[:50]  # cap at 50 rows to avoid context overflow
            return json.dumps(rows, indent=2, default=str)
        except Exception as e:
            return f"SQL ERROR: {str(e)}"

    else:
        return f"Unknown tool: {tool_name}"


# ── Rate limit retry wrapper ─────────────────────────────────────────

def _send_with_retry(conversation, message, max_retries: int = 5):
    """
    Send a message with exponential backoff on rate limit errors.

    On 429: extracts suggested wait time from error if available,
    falls back to (attempt+1) * 30 seconds.
    After max_retries, raises the original exception.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return conversation.send_message(message)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                last_error = e
                wait       = (attempt + 1) * 30
                try:
                    match = re.search(r'retry[^\d]+(\d+)', str(e))
                    if match:
                        wait = int(match.group(1)) + 5
                except Exception:
                    pass
                logger.warning(
                    f"Rate limit hit — waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise e

    raise last_error


# ── Response helpers ─────────────────────────────────────────────────

def _has_tool_call(response) -> bool:
    """Check if the model response contains a tool call."""
    return (
        bool(response.candidates)
        and bool(response.candidates[0].content.parts)
        and any(
            hasattr(p, "function_call") and p.function_call.name
            for p in response.candidates[0].content.parts
        )
    )


# Phrases that indicate the model is mid-answer, not done
_TRANSITION_PHRASES = [
    "let's look at",
    "let me now",
    "now let's",
    "i will now",
    "let me retrieve",
    "let me check",
    "i'll now",
    "moving on to",
    "next, let",
]

def _has_text_answer(response) -> bool:
    """
    Check if the model has a complete final answer.
    Filters out transitional phrases that indicate more work is needed.
    """
    for part in response.candidates[0].content.parts if (
        response.candidates
        and response.candidates[0].content.parts
    ) else []:
        if hasattr(part, "text") and part.text:
            text = part.text.strip().lower()
            if len(text) > 100:
                # Check it's not a transition to more tool calls
                is_transition = any(phrase in text for phrase in _TRANSITION_PHRASES)
                if not is_transition:
                    return True
    return False


def _extract_text(response) -> str:
    """Extract and join all text parts from a model response."""
    parts = []
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            parts.append(part.text)
    return "".join(parts)


# ── Agent ────────────────────────────────────────────────────────────

class PitWallAgent:
    """
    Agentic F1 analytics assistant.

    Design principles:
    - Agent stops naturally when it has a complete answer (>100 chars)
    - MAX_AGENT_ROUNDS in .env is a safety guardrail only
    - Model configurable via GEMINI_MODEL in .env — no code changes to switch
    - Adding tools: FunctionDeclaration + _execute_tool handler only
    - Rate limits handled transparently with exponential backoff
    - Stateless per query — safe for concurrent Streamlit sessions
    """

    def __init__(self, model_name: str = None):
        model_name      = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model_name = model_name
        self.model      = genai.GenerativeModel(
            model_name         = model_name,
            system_instruction = SYSTEM_PROMPT,
            tools              = TOOLS,
        )
        logger.info(f"PitWall agent initialised — model: {model_name}")

    def ask(
        self,
        question:   str,
        max_rounds: int  = None,
        verbose:    bool = False,
    ) -> dict:
        """
        Ask PitWall a question.

        Agent loop terminates when:
          (a) Model produces a complete text answer (>100 chars)  ← natural stop
          (b) Model requests no more tool calls                   ← natural stop
          (c) MAX_AGENT_ROUNDS ceiling hit                        ← safety guardrail

        Returns dict: answer, sql, tool_calls, sources, model, rounds
        """
        max_rounds   = max_rounds or int(os.getenv("MAX_AGENT_ROUNDS", "20"))
        conversation = self.model.start_chat(enable_automatic_function_calling=False)
        tool_calls   = []
        sql_used     = None
        sources_used = set()
        rounds_taken = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Q     : {question}")
            print(f"Model : {self.model_name}")
            print(f"{'='*60}")

        response = _send_with_retry(conversation, question)

        for round_num in range(max_rounds):
            rounds_taken = round_num + 1

            # Natural stop 1 — model has a complete answer
            if _has_text_answer(response):
                if verbose:
                    print(f"\n[Round {round_num}] Model has answer — stopping")
                break

            # Natural stop 2 — no more tool calls requested
            if not _has_tool_call(response):
                if verbose:
                    print(f"\n[Round {round_num}] No tool calls — stopping")
                break

            # Process all tool calls in this round
            tool_results = []

            for part in response.candidates[0].content.parts:
                if not hasattr(part, "function_call") or not part.function_call.name:
                    continue

                tool_name = part.function_call.name
                tool_args = dict(part.function_call.args)

                if verbose:
                    print(f"\n[Round {round_num + 1}] Tool : {tool_name}")
                    print(f"  Args: {json.dumps(tool_args, indent=2)}")

                result = _execute_tool(tool_name, tool_args)

                tool_calls.append({
                    "tool":   tool_name,
                    "args":   tool_args,
                    "result": (str(result)[:200] + "...") if len(str(result)) > 200 else result,
                })

                if tool_name == "run_sql":
                    sql_used = tool_args.get("query")
                    sources_used.add("BigQuery")
                elif tool_name == "get_schema_context":
                    sources_used.add("Schema index")
                elif tool_name == "get_narrative_context":
                    sources_used.add("Wikipedia RAG")

                if verbose:
                    print(f"  Result preview: {str(result)[:300]}...")

                tool_results.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name     = tool_name,
                            response = {"result": str(result)},
                        )
                    )
                )

            response = _send_with_retry(conversation, tool_results)

        else:
            logger.warning(
                f"MAX_AGENT_ROUNDS ({max_rounds}) hit for: '{question}'. "
                f"Increase MAX_AGENT_ROUNDS in .env if needed."
            )

        final_answer = _extract_text(response)
        if not final_answer:
            final_answer = (
                "I was unable to generate an answer. "
                "Please try rephrasing the question."
            )

        if verbose:
            print(f"\nANSWER:\n{final_answer}")
            print(f"\nSOURCES : {list(sources_used)}")
            print(f"TOOLS   : {[t['tool'] for t in tool_calls]}")
            print(f"ROUNDS  : {rounds_taken}")
            if sql_used:
                print(f"\nSQL:\n{sql_used}")

        return {
            "answer":     final_answer,
            "sql":        sql_used,
            "tool_calls": tool_calls,
            "sources":    list(sources_used),
            "model":      self.model_name,
            "rounds":     rounds_taken,
        }


# ── Singleton for Streamlit + evaluator ─────────────────────────────

_agent = None

def get_agent() -> PitWallAgent:
    """
    Singleton — reuses the same agent across Streamlit reruns.
    Model reads from GEMINI_MODEL in .env.
    Switch model: update .env and restart — zero code changes.
    """
    global _agent
    if _agent is None:
        _agent = PitWallAgent()
    return _agent


def ask(question: str, verbose: bool = False) -> dict:
    """
    Module-level convenience function.

    Usage:
        from src.agent import ask
        result = ask("Who has the most wins?")
        print(result["answer"])
    """
    return get_agent().ask(question, verbose=verbose)


# ── Test harness ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "Who has the most race wins in F1 history?",
        "Which team has the fastest average pit stop time?",
        "Tell me about Senna's wet weather driving and show me his win stats",
        "Who has the highest glamour index but worst recent performance?",
    ]

    agent = PitWallAgent()

    for question in test_questions:
        result = agent.ask(question, verbose=True)
        print(f"\n{'='*60}")
        print(f"ANSWER:\n{result['answer']}")
        if result["sql"]:
            print(f"\nSQL:\n{result['sql']}")
        print(f"\nSOURCES    : {result['sources']}")
        print(f"TOOLS USED : {[t['tool'] for t in result['tool_calls']]}")
        print(f"MODEL      : {result['model']}")
        print(f"ROUNDS     : {result['rounds']}")
        input("\nPress Enter for next question...")