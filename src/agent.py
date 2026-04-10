import os
import json
import time
import logging
import re
from typing import Any
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic as anthropic_sdk

from src.bigquery_client import run_query
from src.schema_retriever import retrieve as schema_retrieve
from src.text_retriever import retrieve as text_retrieve

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID   = os.getenv("BIGQUERY_PROJECT_ID")
DATASET      = os.getenv("BIGQUERY_DATASET")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

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

STRICT RULES:
- Always call get_schema_context before writing any SQL — never guess column names
- After calling get_schema_context, you MUST follow up with run_sql
- For hybrid questions call BOTH get_narrative_context AND get_schema_context, then run_sql
- Only give a final answer when you have retrieved context AND executed SQL
- Always use fully qualified table names: `{PROJECT_ID}.{DATASET}.table_name`
- Handle NULL values with COALESCE or IS NOT NULL filters
- The glamour_index table has driver brand value, social following, luxury partnerships
- Synthesise a clear, engaging English answer with a formatted table where appropriate
- Cite which tables and sources you used
- If SQL returns no results, explain why and try an alternative
"""

# ── Gemini tool definitions ──────────────────────────────────────────
GEMINI_TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_schema_context",
                description=(
                    "Retrieves the most relevant BigQuery table schemas for a question. "
                    "Always call this before writing SQL to get accurate table and column names."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "question": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The user question or planned SQL",
                        )
                    },
                    required=["question"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="get_narrative_context",
                description=(
                    "Retrieves relevant text from F1 driver and constructor Wikipedia bios. "
                    "Use for personality, career, history, or 'why'/'who' questions."
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
                    "Executes a SQL query on BigQuery and returns results. "
                    "Only call after get_schema_context to ensure correct column names."
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

# ── Claude tool definitions (Anthropic format) ───────────────────────
CLAUDE_TOOLS = [
    {
        "name": "get_schema_context",
        "description": (
            "Retrieves the most relevant BigQuery table schemas for a question. "
            "Always call this before writing SQL to get accurate table and column names. "
            "Returns table descriptions, columns, and example questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user question or planned SQL"}
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_narrative_context",
        "description": (
            "Retrieves relevant text from F1 driver and constructor Wikipedia biographies. "
            "Use for personality, career history, rivalries, or any 'why'/'who' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to search for"}
            },
            "required": ["question"],
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Executes a SQL query on BigQuery and returns results. "
            "Only call after get_schema_context to ensure correct column names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Valid BigQuery SQL query"}
            },
            "required": ["query"],
        },
    },
]


# ── Tool executor ────────────────────────────────────────────────────
# Shared across both providers — same tool logic regardless of LLM.
# To add a new tool: add elif here + entry in both GEMINI_TOOLS and CLAUDE_TOOLS.

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
            return json.dumps(rows[:50], indent=2, default=str)
        except Exception as e:
            return f"SQL ERROR: {str(e)}"

    else:
        return f"Unknown tool: {tool_name}"


# ── Provider implementations ─────────────────────────────────────────
# Each provider wraps one LLM API behind a common interface:
#   start_chat() → chat object
#   send(chat, message) → response
#   extract_tool_calls(response) → list of {name, args, tool_use_id}
#   extract_text(response) → str
#   has_text_answer(response) → bool
#   make_tool_result(name, result, id) → provider-specific format

class GeminiProvider:
    """
    Wraps Gemini 2.x with function calling.
    Uses genai.protos format required by google-generativeai v0.7.x.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model      = genai.GenerativeModel(
            model_name         = model_name,
            system_instruction = SYSTEM_PROMPT,
            tools              = GEMINI_TOOLS,
        )

    def start_chat(self):
        return self.model.start_chat(enable_automatic_function_calling=False)

    def send(self, chat, message):
        """Send with exponential backoff on rate limits."""
        last_error = None
        for attempt in range(5):
            try:
                return chat.send_message(message)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    last_error = e
                    wait       = (attempt + 1) * 30
                    try:
                        m = re.search(r'retry[^\d]+(\d+)', str(e))
                        if m:
                            wait = int(m.group(1)) + 5
                    except Exception:
                        pass
                    logger.warning(f"Gemini rate limit — waiting {wait}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    raise e
        raise last_error

    def extract_tool_calls(self, response) -> list[dict]:
        calls = []
        if not (response.candidates and response.candidates[0].content.parts):
            return calls
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                calls.append({
                    "name":        part.function_call.name,
                    "args":        dict(part.function_call.args),
                    "tool_use_id": None,
                })
        return calls

    def extract_text(self, response) -> str:
        parts = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
        return "".join(parts)

    def has_text_answer(self, response) -> bool:
        text = self.extract_text(response).strip().lower()
        if len(text) <= 100:
            return False
        transitions = ["let's look at", "let me now", "now let's", "i will now",
                       "let me retrieve", "let me check", "i'll now", "moving on to"]
        return not any(phrase in text for phrase in transitions)

    def make_tool_result(self, tool_name: str, result: str, tool_use_id=None):
        return genai.protos.Part(
            function_response=genai.protos.FunctionResponse(
                name     = tool_name,
                response = {"result": result},
            )
        )


class ClaudeProvider:
    """
    Wraps Anthropic Claude with tool use.
    Uses claude-sonnet-4-5 by default.
    Maintains conversation history as a messages list per Anthropic's stateless API.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client     = anthropic_sdk.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.messages   = []

    def start_chat(self):
        """Reset conversation history for a new query."""
        self.messages = []
        return self

    def send(self, chat, message):
        """
        Send a message to Claude.
        message = str for initial question
        message = list of tool results for subsequent rounds
        """
        if isinstance(message, str):
            self.messages.append({"role": "user", "content": message})
        else:
            # Tool results from previous round
            tool_results = [
                {
                    "type":        "tool_result",
                    "tool_use_id": tr["tool_use_id"],
                    "content":     tr["result"],
                }
                for tr in message
            ]
            self.messages.append({"role": "user", "content": tool_results})

        last_error = None
        for attempt in range(5):
            try:
                response = self.client.messages.create(
                    model      = self.model_name,
                    max_tokens = 2048,
                    system     = SYSTEM_PROMPT,
                    tools      = CLAUDE_TOOLS,
                    messages   = self.messages,
                )
                # Append assistant turn to history for next round
                self.messages.append({"role": "assistant", "content": response.content})
                return response
            except Exception as e:
                if "rate" in str(e).lower() or "529" in str(e) or "overload" in str(e).lower():
                    last_error = e
                    wait       = (attempt + 1) * 30
                    logger.warning(f"Claude rate limit — waiting {wait}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    raise e
        raise last_error

    def extract_tool_calls(self, response) -> list[dict]:
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append({
                    "name":        block.name,
                    "args":        block.input,
                    "tool_use_id": block.id,
                })
        return calls

    def extract_text(self, response) -> str:
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    def has_text_answer(self, response) -> bool:
        return response.stop_reason == "end_turn" and len(self.extract_text(response).strip()) > 100

    def make_tool_result(self, tool_name: str, result: str, tool_use_id=None):
        return {"tool_use_id": tool_use_id, "result": result}


# ── Provider factory ─────────────────────────────────────────────────

def get_provider(model_name: str = None):
    """
    Return the correct LLM provider based on LLM_PROVIDER env var.

    LLM_PROVIDER=gemini (default) → GeminiProvider
    LLM_PROVIDER=claude            → ClaudeProvider

    To add a new provider:
      1. Implement a class with the same interface above
      2. Add an elif here
      No other changes needed anywhere.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "claude":
        m = model_name if (model_name and "claude" in model_name) else "claude-sonnet-4-5"
        logger.info(f"Provider: Claude — model: {m}")
        return ClaudeProvider(m)

    else:  # gemini (default)
        m = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        logger.info(f"Provider: Gemini — model: {m}")
        return GeminiProvider(m)


# ── Agent ────────────────────────────────────────────────────────────

class PitWallAgent:
    """
    Agentic F1 analytics assistant.

    Design principles:
    - Model-agnostic: swap LLM by setting LLM_PROVIDER in .env, zero code changes
    - Agent stops naturally when it has a complete answer — not on arbitrary count
    - MAX_AGENT_ROUNDS is a safety guardrail only (circuit breaker)
    - Adding tools: _execute_tool + provider tool definitions — nothing else changes
    - Stateless per query — safe for concurrent Streamlit sessions
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.provider   = get_provider(model_name)

    def ask(
        self,
        question:   str,
        max_rounds: int  = None,
        verbose:    bool = False,
    ) -> dict:
        """
        Ask PitWall a question.

        Terminates when:
          (a) Provider signals a complete text answer   ← natural stop
          (b) No more tool calls requested              ← natural stop
          (c) MAX_AGENT_ROUNDS ceiling hit              ← safety guardrail

        Returns: answer, sql, tool_calls, sources, model, provider, rounds
        """
        max_rounds   = max_rounds or int(os.getenv("MAX_AGENT_ROUNDS", "20"))
        chat         = self.provider.start_chat()
        tool_calls   = []
        sql_used     = None
        sources_used = set()
        rounds_taken = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Q        : {question}")
            print(f"Provider : {os.getenv('LLM_PROVIDER','gemini')}")
            print(f"{'='*60}")

        response = self.provider.send(chat, question)

        for round_num in range(max_rounds):
            rounds_taken = round_num + 1

            # Natural stop 1 — complete answer
            if self.provider.has_text_answer(response):
                if verbose:
                    print(f"\n[Round {round_num}] Answer ready — stopping")
                break

            # Natural stop 2 — no more tool calls
            calls = self.provider.extract_tool_calls(response)
            if not calls:
                if verbose:
                    print(f"\n[Round {round_num}] No tool calls — stopping")
                break

            # Execute all tool calls in this round
            tool_results = []
            for call in calls:
                tool_name   = call["name"]
                tool_args   = call["args"]
                tool_use_id = call.get("tool_use_id")

                if verbose:
                    print(f"\n[Round {round_num+1}] Tool: {tool_name}")
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
                    print(f"  Result preview: {str(result)[:250]}...")

                tool_results.append(
                    self.provider.make_tool_result(tool_name, str(result), tool_use_id)
                )

            response = self.provider.send(chat, tool_results)

        else:
            logger.warning(f"MAX_AGENT_ROUNDS ({max_rounds}) hit for: '{question}'")

        final_answer = self.provider.extract_text(response)
        if not final_answer:
            final_answer = "I was unable to generate an answer. Please try rephrasing."

        if verbose:
            print(f"\nANSWER:\n{final_answer}")
            print(f"SOURCES  : {list(sources_used)}")
            print(f"ROUNDS   : {rounds_taken}")
            if sql_used:
                print(f"SQL:\n{sql_used}")

        return {
            "answer":     final_answer,
            "sql":        sql_used,
            "tool_calls": tool_calls,
            "sources":    list(sources_used),
            "model":      self.model_name,
            "provider":   os.getenv("LLM_PROVIDER", "gemini"),
            "rounds":     rounds_taken,
        }


# ── Singleton for Streamlit + evaluator ─────────────────────────────

_agent = None

def get_agent() -> PitWallAgent:
    """
    Singleton — reuses agent across Streamlit reruns.
    Provider and model read from .env — zero code changes to switch.
    """
    global _agent
    if _agent is None:
        _agent = PitWallAgent()
    return _agent


def ask(question: str, verbose: bool = False) -> dict:
    """
    Module-level convenience function.
    Usage: from src.agent import ask
    """
    return get_agent().ask(question, verbose=verbose)


# ── Test harness ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Provider: {os.getenv('LLM_PROVIDER', 'gemini')}")

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
        print(f"ANSWER   :\n{result['answer']}")
        print(f"PROVIDER : {result['provider']}")
        print(f"ROUNDS   : {result['rounds']}")
        if result["sql"]:
            print(f"SQL:\n{result['sql']}")
        input("\nPress Enter for next question...")