"""agent.py - the LangGraph agent.

        START
          |
       [router]  <- LLM decides: is this a book question, or a request for reading help?
        /     \
 [answer]   [schedule_propose]   <- propose a day/time (LLM)
     |            |
     |      [schedule_confirm]   <- INTERRUPT: pause, ask the human, then write schedule.json
     |            |
    END          END

The one "tool" is saving a reading session. The agentic decision is the router choosing it.
"""
import json
from datetime import datetime, timedelta
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import ingest
import schedule_store
from llm import CHAT_MODEL, client


class AgentState(TypedDict, total=False):
    """Data that flows through the graph; each node returns the keys it wants to update."""

    question: str  # what the user typed
    book_title: str
    route: Literal["answer", "schedule"]
    reply: str  # text shown to the user at the end
    proposal: dict  # {"date": "YYYY-MM-DD", "time": "HH:MM"}


def _chat(system: str, user: str, json_mode: bool = False) -> str:
    """One chat call. temperature=0 keeps routing/parsing deterministic."""
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------- router
ROUTER_PROMPT = """You route messages for a reading companion app. Reply with exactly one word:

schedule - the user wants help building a reading habit: staying consistent, being reminded,
           keeping track of where they are, setting aside reading time.
answer   - anything else, especially questions about the book's content (plot, characters,
           facts, themes, quotes, summaries).

Examples:
"I keep forgetting to read, can you help me stay consistent?" -> schedule
"Remind me to keep reading" -> schedule
"I keep losing track of where I am" -> schedule
"Who is the main character?" -> answer
"What does chapter 2 say about storms?" -> answer"""


def router_node(state: AgentState) -> dict:
    """Decide which path to take. Anything unexpected falls back to 'answer' (the safe path)."""
    decision = _chat(ROUTER_PROMPT, state["question"]).lower()
    return {"route": "schedule" if decision.startswith("schedule") else "answer"}


def pick_path(state: AgentState) -> str:
    """Conditional edge: the name of the next node, read from the router's decision."""
    return "schedule_propose" if state["route"] == "schedule" else "answer"


# ---------------------------------------------------------------- answer path (RAG)
ANSWER_PROMPT = """You are a reading companion. Answer the user's question using ONLY the book
excerpts provided. If the excerpts don't contain the answer, say you couldn't find it in the book.
Be concise."""


def answer_node(state: AgentState) -> dict:
    """Retrieve the most relevant chunks, put them in the prompt, and let the LLM answer."""
    chunks = ingest.retrieve(state["book_title"], state["question"], k=4)
    if not chunks:  # nothing scored above MIN_SCORE: don't let the LLM guess
        return {"reply": "I couldn't find anything about that in the book."}
    excerpts = "\n\n".join(f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(chunks))
    user_prompt = f'Book: "{state["book_title"]}"\n\n{excerpts}\n\nQuestion: {state["question"]}'
    return {"reply": _chat(ANSWER_PROMPT, user_prompt)}


# ---------------------------------------------------------------- schedule path (tool + human approval)
def schedule_propose_node(state: AgentState) -> dict:
    """Turn the user's wording ("tomorrow at 7pm") into a concrete date + time.

    This is its own node, separate from the confirm step, on purpose: when a LangGraph node
    resumes after an interrupt it re-runs from its first line. Keeping the LLM call here means
    it is NOT repeated (and can't change its mind) when the user answers.
    """
    now = datetime.now()
    system = (
        f"Today is {now:%A, %Y-%m-%d}, and it is {now:%H:%M}. Extract the reading time the user "
        'wants. Reply as JSON: {"date": "YYYY-MM-DD", "time": "HH:MM"} (24-hour). '
        "If they give no day or time, use tomorrow at 19:00."
    )
    try:
        proposal = json.loads(_chat(system, state["question"], json_mode=True))
        datetime.strptime(f'{proposal["date"]} {proposal["time"]}', "%Y-%m-%d %H:%M")  # validate
    except (ValueError, KeyError):
        proposal = {"date": (now + timedelta(days=1)).strftime("%Y-%m-%d"), "time": "19:00"}
    return {"proposal": {"date": proposal["date"], "time": proposal["time"]}}


def schedule_confirm_node(state: AgentState) -> dict:
    """Human-in-the-loop: pause the graph and wait for the user before touching schedule.json."""
    p = state["proposal"]
    # interrupt() freezes the graph and hands this payload to the caller. The graph resumes when
    # the caller sends Command(resume=<answer>), and interrupt() then *returns* that answer.
    confirmed = interrupt({"book": state["book_title"], "date": p["date"], "time": p["time"]})

    if not confirmed:
        return {"reply": "No problem, I haven't scheduled anything."}
    schedule_store.add_session(state["book_title"], p["date"], p["time"])  # the one tool call
    return {"reply": f'Done! Reading session for "{state["book_title"]}" saved for {p["date"]} at {p["time"]}.'}


# ---------------------------------------------------------------- wiring
def build_graph():
    """Assemble the graph. A checkpointer is required for interrupt(): it stores the paused state."""
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("answer", answer_node)
    g.add_node("schedule_propose", schedule_propose_node)
    g.add_node("schedule_confirm", schedule_confirm_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", pick_path, ["answer", "schedule_propose"])
    g.add_edge("answer", END)
    g.add_edge("schedule_propose", "schedule_confirm")
    g.add_edge("schedule_confirm", END)
    return g.compile(checkpointer=MemorySaver())
