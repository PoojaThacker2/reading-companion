"""main.py - terminal chat loop.  Usage: python main.py sample_book.pdf

Starts the FastAPI backend in a background thread, uploads the book, then chats with it over HTTP.
This file contains no AI logic - it only prints and asks for input.
"""
import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv

load_dotenv()

HOST = "127.0.0.1"


def _free_port() -> int:
    """Ask the OS for an unused port so we never collide with another local server."""
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


PORT = _free_port()
BASE = f"http://{HOST}:{PORT}"

ROUTE_LABELS = {
    "answer": "→ routing: answer from book",
    "schedule": "→ routing: schedule a reading session (needs confirmation)",
}


def start_backend() -> None:
    """Run the API server in a daemon thread and wait until it responds."""
    import api  # imported here so the OPENAI_API_KEY check in main() happens first

    config = uvicorn.Config(api.app, host=HOST, port=PORT, log_level="warning")
    threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()
    for _ in range(50):
        try:
            httpx.get(f"{BASE}/docs", timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.2)
    sys.exit("Backend failed to start")


def post(path: str, **kwargs) -> dict:
    """POST to the backend; exit with a readable message on errors."""
    resp = httpx.post(f"{BASE}{path}", timeout=300, **kwargs)
    if resp.status_code != 200:
        sys.exit(f"Error from backend: {resp.text}")
    return resp.json()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which lacks "→"
    if len(sys.argv) != 2:
        sys.exit("Usage: python main.py <book.pdf|book.txt>")
    book = Path(sys.argv[1])
    if not book.exists():
        sys.exit(f"File not found: {book}")
    if not os.getenv("OLLAMA_URL") and not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (or OLLAMA_URL for local models) in a .env file (see .env.example)")

    start_backend()

    print(f"Ingesting {book.name} (chunking + embedding)...")
    with book.open("rb") as f:
        info = post("/ingest", files={"file": (book.name, f)})
    print(f'Loaded "{info["title"]}" ({info["chunks"]} chunks). Ask me anything, or type "quit".\n')

    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in ("quit", "exit"):
            break
        if not message:
            continue

        result = post("/chat", json={"message": message})
        print(ROUTE_LABELS[result["route"]])  # make the agent's decision visible

        if result["status"] == "needs_confirmation":
            p = result["proposal"]
            answer = input(f'Schedule "{p["book"]}" for {p["date"]} at {p["time"]}? [y/n] ').strip().lower()
            result = post("/confirm", json={"confirmed": answer in ("y", "yes")})

        print(f"agent> {result['reply']}\n")


if __name__ == "__main__":
    main()
