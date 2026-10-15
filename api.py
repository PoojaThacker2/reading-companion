"""api.py - thin FastAPI layer over the agent. The terminal UI (main.py) only talks to these routes.

POST /ingest    upload a PDF/TXT; chunk, embed and store it
POST /chat      send a message; returns the route taken and either a reply or a confirmation request
POST /confirm   answer a pending confirmation (yes/no); returns the final reply
"""
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from langgraph.types import Command
from pydantic import BaseModel

import ingest
from agent import build_graph

app = FastAPI(title="Reading Companion")
graph = build_graph()

# One book, one conversation per server run (no accounts, no library).
session = {"book_title": None, "thread_id": str(uuid.uuid4())}


class ChatRequest(BaseModel):
    message: str


class ConfirmRequest(BaseModel):
    confirmed: bool


def _config() -> dict:
    """LangGraph uses thread_id to find this conversation's saved (possibly paused) state."""
    return {"configurable": {"thread_id": session["thread_id"]}}


def _format(result: dict) -> dict:
    """Convert a graph result into JSON: either a pending confirmation or a finished reply."""
    route = result.get("route")
    if "__interrupt__" in result:  # graph is paused at interrupt()
        return {"status": "needs_confirmation", "route": route, "proposal": result["__interrupt__"][0].value}
    return {"status": "done", "route": route, "reply": result["reply"]}


@app.post("/ingest")
def ingest_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(400, "Only .pdf and .txt files are supported")
    title = Path(file.filename).stem.replace("_", " ").replace("-", " ").title()

    with tempfile.TemporaryDirectory() as tmp:  # save the upload so the loaders can read it
        saved = Path(tmp) / file.filename
        with saved.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            n_chunks = ingest.ingest_book(saved, title)
        except ValueError as e:
            raise HTTPException(400, str(e))

    session["book_title"] = title
    return {"title": title, "chunks": n_chunks}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not session["book_title"]:
        raise HTTPException(400, "Ingest a book first")
    state_in = {"question": req.message, "book_title": session["book_title"]}
    return _format(graph.invoke(state_in, _config()))


@app.post("/confirm")
def confirm_endpoint(req: ConfirmRequest):
    # Command(resume=...) wakes the paused graph; the value becomes interrupt()'s return value.
    return _format(graph.invoke(Command(resume=req.confirmed), _config()))
